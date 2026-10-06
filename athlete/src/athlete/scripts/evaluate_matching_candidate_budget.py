"""Evaluate candidate-budget success using the training root-directed sampler.

Paired group tests share physics within each group and launches across all
libraries. Count raw candidates returned by the training sampler, not a bank
pre-filtered for successful net clearance, landing location or matching.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, default=Path("artifacts/matching_scaling_020_20260912_final"))
    parser.add_argument("--groups", type=int, default=10000)
    parser.add_argument("--group-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    with (out / "imports.log").open("w") as log, contextlib.redirect_stdout(log):
        from athlete.goal_cond_tracking.torch_tennis_planner import (
            sample_root_directed_tennis_launches_torch,
            match_tennis_trajectories_to_motions_torch,
        )
        from athlete.goal_cond_tracking.warp_tennis_planner import simulate_tennis_trajectories_warp_fused
        from athlete.scripts.tennis_physics import STANDARD_TENNIS_DOMAIN_RANDOMIZATION as dr

    manifest = json.loads((args.source_results / "motion_manifest.json").read_text())
    subset_names = json.loads((args.source_results / "subsets.json").read_text())
    index = {m["name"]: m["index"] for m in manifest}
    targets = torch.tensor([m["target_xyz_m"] for m in manifest], device=device)
    nominal = torch.tensor([m["nominal_contact_time_s"] for m in manifest], device=device)
    subsets = {key: torch.tensor([index[n] for n in names], device=device) for key, names in subset_names.items()}
    budgets = [1, 2, 4, 8, 16, 24]
    sizes = [100, 200, 500, 1000]
    seeds = [0, 1, 2]
    kmax = max(budgets)
    # These are the base RootDirected-Cache0-200 config values, not SmallCourt.
    low = torch.tensor([8.0, -2.0, 0.5], device=device)
    high = torch.tensor([10.0, 2.0, 1.3], device=device)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    root = torch.zeros((args.groups * kmax, 2), device=device)
    p0, v0 = sample_root_directed_tennis_launches_torch(
        root, launch_position_min=low, launch_position_max=high,
        horizontal_speed_range_m_s=(3.5, 5.25), horizontal_angle_half_width_deg=15.0,
        net_crossing_height_range_m=(1.5, 3.5), maximum_initial_speed_m_s=7.0,
        net_x_m=5.6,
    )
    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(args.groups, device=device)
    physics = {"mass": uniform(dr.ball_mass_kg), "restitution": uniform(dr.court_restitution),
               "tangent": uniform(dr.ground_tangent_speed_retention), "drag": uniform(dr.drag_coefficient)}
    np.savez_compressed(out / "candidate_launches.npz", positions=p0.cpu().numpy().reshape(args.groups,kmax,3),
                        velocities=v0.cpu().numpy().reshape(args.groups,kmax,3),
                        **{key: value.cpu().numpy() for key,value in physics.items()})
    distances = {key: np.empty((args.groups,kmax), dtype=np.float32) for key in subsets}
    net_clear = np.empty((args.groups,kmax),dtype=bool)
    post_bounce = np.empty_like(net_clear)
    for lo in range(0,args.groups,args.group_batch):
        hi = min(args.groups,lo+args.group_batch)
        with torch.no_grad():
            traj = simulate_tennis_trajectories_warp_fused(
                p0[lo*kmax:hi*kmax], v0[lo*kmax:hi*kmax],
                ball_mass_kg=physics["mass"][lo:hi].repeat_interleave(kmax),
                court_restitution=physics["restitution"][lo:hi].repeat_interleave(kmax),
                tangent_speed_retention=physics["tangent"][lo:hi].repeat_interleave(kmax),
                drag_coefficient=physics["drag"][lo:hi].repeat_interleave(kmax),
                dt=0.01,horizon_s=4.0,net_x=5.6,net_half_width=5.485,
            )
            net_clear[lo:hi] = traj.net_cleared.cpu().numpy().reshape(-1,kmax)
            post_bounce[lo:hi] = (traj.bounce_counts==1).any(1).cpu().numpy().reshape(-1,kmax)
            for key,ids in subsets.items():
                if key.endswith("_n1000") and key!="seed0_n1000":
                    distances[key][lo:hi] = distances["seed0_n1000"][lo:hi]
                    continue
                match = match_tennis_trajectories_to_motions_torch(
                    traj,targets[ids],nominal[ids]/2,nominal[ids],maximum_distance=.2,
                    motion_chunk_size=200,contact_time_offset_s=.01,
                    hierarchical_top_k=32,hierarchical_coarse_neighbor_radius=1,
                )
                dist = match.distances.cpu().numpy()
                np.testing.assert_array_equal(match.valid.cpu().numpy(),dist<=.2)
                distances[key][lo:hi] = dist.reshape(-1,kmax)
        if lo//args.group_batch%10==0 or hi==args.groups:
            print(f"[match] {hi}/{args.groups} groups x {kmax} candidates; {time.perf_counter()-start:.1f}s",flush=True)
    np.savez_compressed(out/"candidate_distances.npz",**distances)
    np.savez_compressed(out/"candidate_masks.npz",net_cleared=net_clear,post_bounce_segment=post_bounce)
    rows=[]; runtime=[]
    for key,distance in distances.items():
        seed=int(key.split('_')[0][4:]);size=int(key.split('_')[1][1:])
        hit=distance<=.2
        previous=np.zeros(args.groups,dtype=bool)
        for budget in budgets:
            success=hit[:,:budget].any(1)
            assert np.all(~previous|success)
            previous=success
            rows.append({"size":size,"subset_seed":seed,"budget":budget,"groups":args.groups,
                         "successful_groups":int(success.sum()),"group_success_rate":float(success.mean()),
                         "mean_raw_candidate_success_rate":float(hit[:,:budget].mean())})
        wave_hit=hit.reshape(args.groups,12,2).any(-1)
        success=wave_hit.any(-1)
        first_wave=wave_hit.argmax(-1)
        # The training 'attempts' return value is the chosen candidate's index;
        # count the whole simulated wave to measure actual candidate consumption.
        consumed=np.where(success,(first_wave+1)*2,kmax)
        selected=np.where(success,first_wave,11)
        wave_dist=distance.reshape(args.groups,12,2).min(-1)
        residual=wave_dist[np.arange(args.groups),selected][success]
        runtime.append({"size":size,"subset_seed":seed,"wave_size":2,"budget":24,
                        "mean_candidates_evaluated_in_early_stop":float(consumed.mean()),
                        "p95_candidates_evaluated_in_early_stop":float(np.percentile(consumed,95)),
                        "successful_groups":int(success.sum()),
                        "mean_selected_distance_m_if_success":float(residual.mean()) if len(residual) else None,
                        "no_finite_match_in_all_24_candidates":int((~np.isfinite(distance).any(1)).sum()),
                        "budget_exhausted_without_success":int((~success).sum())})
    summary=[]
    for size in sizes:
        for budget in budgets:
            group=[r for r in rows if r['size']==size and r['budget']==budget]
            values=np.array([r['group_success_rate'] for r in group])
            summary.append({"size":size,"budget":budget,"rate_mean":float(values.mean()),
                            "rate_subset_std":float(values.std(ddof=1)) if size<1000 else 0.0,
                            "counts_seed_0_1_2":[r['successful_groups'] for r in group]})
    for filename,records in [('per_seed.csv',rows),('summary.csv',summary),('adaptive_two_candidate_waves.csv',runtime)]:
        with (out/filename).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    dump(out/'summary.json',summary)
    dump(out/'protocol.json',{
        "groups":args.groups,"max_candidates_per_group":kmax,"budgets":budgets,"sizes":sizes,
        "query_seed":args.seed,"subset_seeds":seeds,"source_results":str(args.source_results.resolve()),
        "root_xy":[0,0],"root_yaw":0,"within_group_physics":"shared across all 24 candidates",
        "sampler":"sample_root_directed_tennis_launches_torch, includes its internal analytic feasibility resampling",
        "position_min":[8,-2,.5],"position_max":[10,2,1.3],"horizontal_speed_range_m_s":[3.5,5.25],
        "horizontal_angle_half_width_deg":15,"analytic_net_crossing_height_range_m":[1.5,3.5],
        "maximum_initial_speed_m_s":7,"dt_s":.01,"horizon_s":4,"net_x_m":5.6,"net_half_width_m":5.485,
        "ball_mass_kg":list(dr.ball_mass_kg),"court_restitution":list(dr.court_restitution),
        "tangent_speed_retention":list(dr.ground_tangent_speed_retention),"drag_coefficient":list(dr.drag_coefficient),
        "threshold_m":.2,"speedup":2,"deadline_offset_s":.01,
        "prefilter":"none; all sampler-returned candidates, including failed matches, are counted",
        "backend":"warp_fused","matcher":"production hierarchical top-32 with neighbor radius 1",
        "wave_size_for_early_stop":2,"exclusions":["planner fallback launches","outer episode-initialization retries","intentional failure trajectories","cache reuse","moving robot states"],
        "net_cleared_flag_fraction":float(net_clear.mean()),"post_first_bounce_segment_fraction":float(post_bounce.mean()),
        "group_batch":args.group_batch,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(device),
        "elapsed_seconds":time.perf_counter()-start,"peak_torch_allocated_mb":torch.cuda.max_memory_allocated(device)/1e6,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation":"Planning success under a finite candidate budget at normalized initial robot pose, not contact success or end-to-end training throughput",
    })
    print(json.dumps(summary,indent=2),flush=True)
    print(f"DONE: {out}",flush=True)


if __name__=='__main__':
    main()
