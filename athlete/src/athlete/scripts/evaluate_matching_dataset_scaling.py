"""Paired offline trajectory-matching evaluation for nested motion libraries.

Reuse the training ball physics and matcher convention, cache one independent
test bank, and evaluate every library against the same trajectories. No robot
policy is trained or executed by this experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from athlete.scripts.matching_reachability import trajectory_reachability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / den
    return center - half, center + half


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-config", type=Path, default=Path(
        "athlete/src/athlete/motion_sets/motion_train_configs/"
        "rollout_1000epis_relabelled_all_phase.toml"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-trajectories", type=int, default=10000)
    parser.add_argument("--query-seed", type=int, default=20260912)
    parser.add_argument("--distance", type=float, default=0.2)
    parser.add_argument("--sample-batch", type=int, default=8192)
    parser.add_argument("--match-batch", type=int, default=128)
    parser.add_argument("--motion-chunk", type=int, default=32)
    parser.add_argument("--max-raw-samples", type=int, default=2000000)
    parser.add_argument("--max-contact-speedup", type=float, default=2.0)
    parser.add_argument("--deadline-offset", type=float, default=0.01)
    parser.add_argument("--eligibility", choices=["first_bounce_disk", "reachable_region"],
                        default="first_bounce_disk",
                        help="Legacy first-bounce disk, or an in-range trajectory-height test")
    parser.add_argument("--reach-radius", type=float, default=3.0,
                        help="Horizontal distance to G1 used by reachable_region (m)")
    parser.add_argument("--minimum-reach-height", type=float, default=0.5,
                        help="Minimum ball-center height attained inside the reach disk (m)")
    parser.add_argument("--bounce-scope", choices=["post_first", "pre_second"], default="post_first",
                        help="Phase used for trajectory eligibility; motion matching remains post_first")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve previous results.")
    if args.distance <= 0 or args.num_trajectories < 1 or args.max_contact_speedup < 1:
        raise ValueError("Invalid evaluation parameters")
    if not math.isfinite(args.reach_radius) or args.reach_radius <= 0:
        raise ValueError("reach-radius must be finite and positive")
    if not math.isfinite(args.minimum_reach_height) or args.minimum_reach_height < 0:
        raise ValueError("minimum-reach-height must be finite and nonnegative")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.query_seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()

    # Package registration logs are verbose; keep them as evidence, not console noise.
    with (out / "project_import.log").open("w") as log, contextlib.redirect_stdout(log):
        from athlete.scripts.evaluate_trajectory_motion_matching import (
            _actual_net_crossing, _load_motion_match_data,
        )
        from athlete.goal_cond_tracking.torch_tennis_planner import (
            TorchTennisTrajectoryBatch, match_tennis_trajectories_to_motions_torch,
            simulate_tennis_trajectories_torch,
        )
        from athlete.motion_sets.motion_set import MotionSet
        from athlete.scripts.tennis_physics import (
            STANDARD_TENNIS_DOMAIN_RANDOMIZATION, STANDARD_TENNIS_PHYSICS,
        )

    config_path = args.motion_config.resolve()
    print("Loading motion targets and auditing the motion pool...", flush=True)
    targets, _, nominal, names = _load_motion_match_data(config_path, device)
    minimum = nominal / args.max_contact_speedup  # Do not use the old hard-coded /2.
    count_m = len(names)
    if max(args.sizes) > count_m:
        raise ValueError(f"Requested {max(args.sizes)} motions but pool contains {count_m}")
    if not torch.isfinite(targets).all() or not torch.isfinite(nominal).all():
        raise ValueError("Non-finite motion targets or contact times")
    manifest = []
    for index, filename in enumerate(MotionSet.from_toml(config_path).local_motion_files):
        path = Path(filename)
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text())
        with np.load(path) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joints = data["joint_pos"]
            if fps != 50 or not np.isfinite(joints).all():
                raise ValueError(f"Invalid frame rate or non-finite joints: {path}")
            if not 0 < int(meta["strike_frame"]) < len(joints):
                raise ValueError(f"Invalid strike frame: {path}")
        manifest.append({
            "index": index, "name": names[index], "path": str(path.resolve()),
            "sha256": sha256(path), "sidecar_sha256": sha256(sidecar),
            "source_clip": meta["clip"], "frames": len(joints), "fps": fps,
            "strike_frame": meta["strike_frame"],
            "target_xyz_m": targets[index].cpu().tolist(),
            "nominal_contact_time_s": float(nominal[index].item()),
        })
    unique_files = len({m["sha256"] for m in manifest})
    if unique_files != count_m:
        raise ValueError(f"Motion pool contains duplicate files: {unique_files}/{count_m}")
    save_json(out / "motion_manifest.json", manifest)
    print(f"Pool: {count_m} unique files, "
          f"{len({m['source_clip'] for m in manifest})} source clips.", flush=True)

    position_min = (8.0, -2.0, 0.5)
    position_max = (10.0, 2.0, 1.3)
    velocity_min = (-5.25, -1.25, 3.0)
    velocity_max = (-3.5, 1.25, 5.0)
    dt, horizon, bounce_radius_limit = 0.01, 4.0, 3.0
    dr = STANDARD_TENNIS_DOMAIN_RANDOMIZATION

    def uniform(lo, hi, n):
        lower = torch.as_tensor(lo, dtype=torch.float32, device=device)
        upper = torch.as_tensor(hi, dtype=torch.float32, device=device)
        shape = (n,) if lower.ndim == 0 else (n, len(lo))
        return lower + (upper - lower) * torch.rand(shape, device=device)

    bank = {name: [] for name in (
        "initial_positions", "initial_velocities", "mass", "restitution",
        "tangent_retention", "drag", "positions", "velocities", "bounce_counts",
        "first_bounce_positions", "raw_indices", "first_bounce_radius",
        "region_max_height", "region_min_xy",
        "post_first_region_max_height", "post_first_region_min_xy",
        "post_first_within_radius", "post_first_has_height",
        "pre_second_region_max_height", "pre_second_region_min_xy",
        "pre_second_within_radius", "pre_second_has_height",
    )}
    accepted = 0
    drawn = 0
    counts = {k: 0 for k in (
        "raw_consumed", "net_clear", "first_bounce", "within_radius", "has_height", "eligible",
        "post_first_within_radius", "post_first_has_height", "post_first_net_eligible",
        "pre_second_within_radius", "pre_second_has_height", "pre_second_net_eligible",
        "legacy_first_bounce_within_radius", "legacy_net_eligible",
    )}
    sample_start = time.perf_counter()
    # Isolate the test-bank RNG from module registration and data loading.
    torch.manual_seed(args.query_seed)
    while accepted < args.num_trajectories:
        if drawn >= args.max_raw_samples:
            raise RuntimeError("Raw sample limit reached without enough eligible trajectories")
        n = min(args.sample_batch, args.max_raw_samples - drawn)
        p0, v0 = uniform(position_min, position_max, n), uniform(velocity_min, velocity_max, n)
        mass = uniform(*dr.ball_mass_kg, n)
        restitution = uniform(*dr.court_restitution, n)
        tangent = uniform(*dr.ground_tangent_speed_retention, n)
        drag = uniform(*dr.drag_coefficient, n)
        traj = simulate_tennis_trajectories_torch(
            p0, v0, ball_mass_kg=mass, court_restitution=restitution,
            tangent_speed_retention=tangent, drag_coefficient=drag, dt=dt, horizon_s=horizon,
        )
        _, clear, _, _, _ = _actual_net_crossing(traj, height_range_m=(1.5, 3.5))
        first = traj.bounce_counts >= 1
        has_first = first.any(dim=1)
        first_idx = first.to(torch.int64).argmax(dim=1)
        bounce_pos = traj.positions[torch.arange(n, device=device), first_idx]
        diagnostics = {
            scope: trajectory_reachability(
                traj.positions, traj.bounce_counts,
                reach_radius=args.reach_radius, minimum_reach_height=args.minimum_reach_height,
                bounce_scope=scope,
            )
            for scope in ("post_first", "pre_second")
        }
        selected = diagnostics[args.bounce_scope]
        legacy_within = has_first & (torch.linalg.vector_norm(bounce_pos[:, :2], dim=1) <= bounce_radius_limit)
        within = legacy_within if args.eligibility == "first_bounce_disk" else selected.within_radius
        eligible = clear & (legacy_within if args.eligibility == "first_bounce_disk" else selected.has_height)
        ids = torch.where(eligible)[0][:args.num_trajectories - accepted]
        consumed = int(ids[-1].item()) + 1 if len(ids) and accepted + len(ids) == args.num_trajectories else n
        counts["raw_consumed"] += consumed
        count_masks = {
            "net_clear": clear, "first_bounce": has_first, "within_radius": within,
            "has_height": selected.has_height, "eligible": eligible,
            "legacy_first_bounce_within_radius": legacy_within,
            "legacy_net_eligible": clear & legacy_within,
        }
        for scope, diag in diagnostics.items():
            count_masks[f"{scope}_within_radius"] = diag.within_radius
            count_masks[f"{scope}_has_height"] = diag.has_height
            count_masks[f"{scope}_net_eligible"] = clear & diag.has_height
        for key, mask in count_masks.items():
            counts[key] += int(mask[:consumed].sum().item())
        values = {
            "initial_positions": p0, "initial_velocities": v0, "mass": mass,
            "restitution": restitution, "tangent_retention": tangent, "drag": drag,
            "positions": traj.positions, "velocities": traj.velocities,
            "bounce_counts": traj.bounce_counts, "first_bounce_positions": bounce_pos,
            "raw_indices": torch.arange(drawn, drawn + n, device=device),
            "first_bounce_radius": selected.first_bounce_radius,
            "region_max_height": selected.region_max_height,
            "region_min_xy": selected.region_min_xy,
        }
        for scope, diag in diagnostics.items():
            for name in ("region_max_height", "region_min_xy", "within_radius", "has_height"):
                values[f"{scope}_{name}"] = getattr(diag, name)
        if len(ids):
            for key, value in values.items():
                bank[key].append(value[ids].cpu().numpy())
        accepted += len(ids)
        drawn += n
        print(f"[sample] {accepted}/{args.num_trajectories} eligible; "
              f"{counts['raw_consumed']} raw consumed; {time.perf_counter()-sample_start:.1f}s", flush=True)

    sample_seconds = time.perf_counter() - sample_start
    bank = {key: np.concatenate(value, axis=0) for key, value in bank.items()}
    bank["times"] = traj.times.cpu().numpy()
    np.savez_compressed(out / "test_trajectories.npz", **bank)
    counts["raw_generated_in_full_batches"] = drawn
    counts["eligible_fraction"] = counts["eligible"] / counts["raw_consumed"]
    save_json(out / "sampling_counts.json", counts)
    del traj, values

    m = args.num_trajectories
    joint = np.empty((m, count_m), dtype=np.float32)
    spatial = np.empty_like(joint)
    times = torch.as_tensor(bank["times"], device=device)
    command_times = times + args.deadline_offset
    gate = (command_times[None, :] >= minimum[:, None]) & (command_times[None, :] <= nominal[:, None])
    match_start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, m, args.match_batch):
            hi = min(lo + args.match_batch, m)
            positions = torch.as_tensor(bank["positions"][lo:hi], device=device)
            post = torch.as_tensor(bank["bounce_counts"][lo:hi], device=device) == 1
            for ml in range(0, count_m, args.motion_chunk):
                mh = min(ml + args.motion_chunk, count_m)
                distance_sq = ((positions[:, None] - targets[None, ml:mh, None]) ** 2).sum(-1)
                distance_sq.masked_fill_(~post[:, None], torch.inf)
                spatial[lo:hi, ml:mh] = distance_sq.min(-1).values.sqrt().cpu().numpy()
                distance_sq.masked_fill_(~gate[None, ml:mh], torch.inf)
                joint[lo:hi, ml:mh] = distance_sq.min(-1).values.sqrt().cpu().numpy()
            if hi == m or lo // args.match_batch % 10 == 0:
                print(f"[match] {hi}/{m} trajectories x {count_m} motions; "
                      f"{time.perf_counter()-match_start:.1f}s", flush=True)
    match_seconds = time.perf_counter() - match_start
    np.save(out / "spatiotemporal_distances.npy", joint)
    np.save(out / "spatial_distances.npy", spatial)

    # Independent cross-check against the existing production matcher.
    check_ids = np.linspace(0, m - 1, min(m, 64), dtype=int)
    check_traj = TorchTennisTrajectoryBatch(
        times=times, positions=torch.as_tensor(bank["positions"][check_ids], device=device),
        velocities=torch.as_tensor(bank["velocities"][check_ids], device=device),
        bounce_counts=torch.as_tensor(bank["bounce_counts"][check_ids], device=device),
        net_cleared=torch.ones(len(check_ids), dtype=torch.bool, device=device),
    )
    permutations = {seed: np.random.default_rng(seed).permutation(count_m) for seed in args.subset_seeds}
    validation = []
    for size in args.sizes:
        idx = permutations[args.subset_seeds[0]][:size]
        ti = torch.as_tensor(idx, device=device)
        result = match_tennis_trajectories_to_motions_torch(
            check_traj, targets[ti], minimum[ti], nominal[ti], maximum_distance=args.distance,
            motion_chunk_size=args.motion_chunk, contact_time_offset_s=args.deadline_offset,
        )
        expected = joint[check_ids][:, idx].min(axis=1)
        actual = result.distances.cpu().numpy()
        np.testing.assert_array_equal(np.isfinite(expected), np.isfinite(actual))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], atol=2e-4, rtol=1e-3)
        np.testing.assert_array_equal(result.valid.cpu().numpy(), expected <= args.distance)
        validation.append({"size": size, "checked_trajectories": len(check_ids),
                           "max_distance_difference_m": float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.0})

    rows, outcomes, subset_manifest = [], {}, {}
    for seed, perm in permutations.items():
        last_hit = np.zeros(m, dtype=bool)
        for size in args.sizes:
            idx = perm[:size]
            distance = joint[:, idx].min(axis=1)
            spatial_distance = spatial[:, idx].min(axis=1)
            hit = distance <= args.distance
            spatial_hit = spatial_distance <= args.distance
            assert np.all(~last_hit | hit), "Nested-library coverage must be monotone"
            assert np.all(~hit | spatial_hit)
            last_hit = hit
            low, high = wilson(int(hit.sum()), m)
            rows.append({
                "size": size, "subset_seed": seed, "test_trajectories": m,
                "matched": int(hit.sum()), "match_rate": float(hit.mean()),
                "spatial_matched": int(spatial_hit.sum()), "spatial_match_rate": float(spatial_hit.mean()),
                "wilson95_low": low, "wilson95_high": high,
                "time_window_loss_pp": float(100 * (spatial_hit.mean() - hit.mean())),
            })
            outcomes[f"seed{seed}_n{size}_hit"] = hit
            outcomes[f"seed{seed}_n{size}_distance"] = distance
            subset_manifest[f"seed{seed}_n{size}"] = [names[i] for i in idx]
    summary = []
    for size in args.sizes:
        group = [r for r in rows if r["size"] == size]
        rates = np.array([r["match_rate"] for r in group])
        srates = np.array([r["spatial_match_rate"] for r in group])
        summary.append({"size": size, "test_trajectories": m,
                        "match_rate_mean": float(rates.mean()),
                        "match_rate_subset_std": float(rates.std(ddof=1)) if len(rates)>1 else 0.0,
                        "spatial_rate_mean": float(srates.mean()),
                        "spatial_rate_subset_std": float(srates.std(ddof=1)) if len(srates)>1 else 0.0,
                        "matched_counts": [r["matched"] for r in group],
                        "subset_seeds": args.subset_seeds})
    for filename, records in [("results_per_seed.csv", rows), ("results_summary.csv", summary)]:
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    np.savez_compressed(out / "per_trajectory_outcomes.npz", **outcomes)
    save_json(out / "subsets.json", subset_manifest)
    save_json(out / "validation.json", validation)
    protocol = {
        "sizes": args.sizes, "subset_seeds": args.subset_seeds, "query_seed": args.query_seed,
        "query_count": m, "threshold_m": args.distance, "motion_pool_count": count_m,
        "unique_source_clips": len({a["source_clip"] for a in manifest}),
        "motion_config": str(config_path), "motion_config_sha256": sha256(config_path),
        "position_min": position_min, "position_max": position_max,
        "velocity_min": velocity_min, "velocity_max": velocity_max,
        "dt_s": dt, "horizon_s": horizon,
        "eligibility": args.eligibility,
        "first_bounce_radius_limit_m": bounce_radius_limit if args.eligibility == "first_bounce_disk" else None,
        "reach_radius_m": args.reach_radius,
        "minimum_reach_height_m": args.minimum_reach_height,
        "bounce_scope": args.bounce_scope,
        "eligibility_description": (
            "Actual incoming net crossing and first-bounce horizontal radius <= 3 m; "
            "height and reach phase diagnostics do not affect this legacy selection"
            if args.eligibility == "first_bounce_disk" else
            "Actual incoming net crossing and at least one sample with horizontal radius "
            f"<= {args.reach_radius:g} m and ball-center height >= {args.minimum_reach_height:g} m "
            + ("after first bounce and before second bounce" if args.bounce_scope == "post_first"
               else "before second bounce, including the pre-first-bounce phase")
        ),
        "reach_region_geometry": "Horizontal XY disk centered at the initial G1 root [0, 0] m",
        "reach_height_definition": "Maximum world ball-center z among samples in both the reach disk and selected bounce phase",
        "reach_sampling": "Discrete trajectory samples at dt_s; inclusive radius and height thresholds",
        "reach_diagnostics": {
            "region_max_height": "Maximum ball-center z inside the disk and selected phase; -inf if empty",
            "region_min_xy": "Minimum XY distance to G1 over the selected phase, including points outside the disk; +inf if no phase",
            "first_bounce_radius": "XY radius at the first sampled bounce_count >= 1; NaN if absent",
            "phase_comparison": "NPZ includes both post_first_* and pre_second_* diagnostics for every accepted query",
        },
        "net_clearance": "Actual crossing within net width, above net, before first bounce",
        "ball_phase": "After first bounce and before second bounce",
        "matching_domain": "All samples after first bounce and before second bounce, without a further reach-disk or height mask; eligibility selects whole incoming trajectories only",
        "time_window": "T/max_contact_speedup <= t+deadline_offset <= T",
        "max_contact_speedup": args.max_contact_speedup, "deadline_offset_s": args.deadline_offset,
        "physics": asdict(STANDARD_TENNIS_PHYSICS), "domain_randomization": asdict(dr),
        "physics_group_size": 1, "sampler": "Independent uniform launch parameters; no motion-conditioned rejection",
        "initial_robot_frame": "Each motion root XY and yaw are normalized; root height/tilt retained",
        "subset_method": "Uniform random permutations and nested prefixes, without replacement",
        "uncertainty": "SD over subset permutations, not independent training/generation runs",
        "evaluation": "Offline discrete point-and-time compatibility; not actual racket contact or policy rollout",
        "dataset": "Existing relabelled rollout pool; no new reference generation or policy training",
        "torch": torch.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "peak_torch_allocated_mb": torch.cuda.max_memory_allocated(device)/1e6 if device.type=="cuda" else None,
        "sample_seconds": sample_seconds, "match_seconds": match_seconds,
        "elapsed_seconds": time.perf_counter()-wall_start,
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__)), "sampling_counts": counts,
        "reachability_script_sha256": sha256(Path(__file__).with_name("matching_reachability.py")),
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "results_summary.json", summary)
    # Diagnostic only: do not mix the existing training subset into the random-size sweep.
    first200 = np.array([i for i,n in enumerate(names) if n < "ep_0200"])
    if len(first200)==200:
        hit = joint[:,first200].min(axis=1)<=args.distance
        save_json(out/"existing_first200_diagnostic.json", {"motion_count":200,"matched":int(hit.sum()),"total":m,"match_rate":float(hit.mean())})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
