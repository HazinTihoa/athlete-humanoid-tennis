"""Render original G1 strike NPZ clips as kinematic replay previews.

Requires mujoco, numpy, Pillow and imageio-ffmpeg. For headless rendering, set
MUJOCO_GL=egl before running. Does not simulate a policy or modify motion data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_matching.build_official_mm_db import (  # noqa: E402
    ISAACLAB_G1_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media/strike-replays")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default(size=24)
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default(size=18)
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/replay_unitree_description/mjcf/g1.xml"))
    spec.worldbody.add_geom(name="preview_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[20, 20, .1], rgba=[.83, .89, .86, 1])
    model = spec.compile()
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).id for name in ISAACLAB_JOINT_NAMES]
    qpos_ids = model.jnt_qposadr[joint_ids]
    body_ids = [model.body(name).id for name in ISAACLAB_G1_BODY_NAMES]
    with np.load(ROOT / "motion_db/g1_racket_34body_augment.npz") as mapping:
        dataset_body_ids = np.zeros(30, dtype=int)
        for base_id, augmented_id in mapping["base_place"]:
            dataset_body_ids[base_id] = augmented_id
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.4, 135, -16
    options = mujoco.MjvOption()
    options.geomgroup[3] = False  # Hide collision geometry.
    renderer = mujoco.Renderer(model, height=height, width=width)
    report = {"type": "kinematic replay of original NPZ data", "playback_speed": 1, "clips": []}
    groups = {"forehand": ["fh_01_g1", "fh_32_g1", "fh_64_g1"],
              "backhand": ["bh_01_g1", "bh_22_g1", "bh_45_g1"]}
    try:
        for group, names in groups.items():
            output = args.output_dir / f"{group}.mp4"
            writer = imageio_ffmpeg.write_frames(str(output), (width, height), fps=50,
                codec="libx264", pix_fmt_out="yuv420p", output_params=["-crf", "23", "-movflags", "+faststart"])
            writer.send(None)
            try:
                for name in names:
                    path = ROOT / "data/djvkovic_npz_data_v5" / f"{name}.npz"
                    with np.load(path) as source:
                        joint_pos = source["joint_pos"]
                        body_pos = source["body_pos_w"]
                        body_quat = source["body_quat_w"]
                        fps = float(source["fps"].reshape(-1)[0])
                    if fps != 50 or joint_pos.shape[1] != len(joint_ids):
                        raise ValueError(f"Unsupported frame rate or joint layout: {path}")
                    strike = json.loads(path.with_suffix(".json").read_text())["strike_frame"]
                    camera.lookat[:] = [*body_pos[:, 0, :2].mean(axis=0), .85]
                    foot_ids = [dataset_body_ids[ISAACLAB_G1_BODY_NAMES.index(name)]
                                for name in ["left_ankle_roll_link", "right_ankle_roll_link"]]
                    floor = float(np.min(body_pos[:, foot_ids, 2])) - .045
                    model.geom_pos[model.geom("preview_ground").id, 2] = floor
                    errors = []
                    for frame in range(len(joint_pos)):
                        data.qpos[:3] = body_pos[frame, 0]
                        data.qpos[3:7] = body_quat[frame, 0]
                        data.qpos[qpos_ids] = joint_pos[frame]
                        mujoco.mj_forward(model, data)
                        errors.append(float(np.sqrt(np.mean((data.xpos[body_ids] - body_pos[frame, dataset_body_ids]) ** 2))))
                        if errors[-1] > .01:
                            raise ValueError(f"FK does not match recorded body positions: {name} frame {frame}")
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        image = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, width, 64), fill="#17352d")
                        draw.text((24, 18), f"{group.title()}  |  {name}.npz", font=font, fill="white")
                        draw.rectangle((0, height-42, width, height), fill="#17352d")
                        draw.text((24, height-32), "Original dataset · Kinematic replay · 1× speed", font=small, fill="#e8efe6")
                        draw.text((width-164, height-32), f"{frame / fps:.2f} / {len(joint_pos) / fps:.2f} s", font=small, fill="#e8efe6")
                        writer.send(np.asarray(image))
                        if frame == strike and name == names[0]:
                            image.save(args.output_dir / f"{group}-poster.jpg")
                    for _ in range(20):
                        writer.send(np.asarray(image))
                    report["clips"].append({"file": str(path.relative_to(ROOT)), "frames": len(joint_pos),
                        "fps": fps, "strike_frame": strike, "body_position_rmse_m": float(np.mean(errors))})
            finally:
                writer.close()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(output),
                "-filter_complex", "fps=12.5,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop", "0", str(args.output_dir / f"{group}.gif")], check=True, capture_output=True)
    finally:
        renderer.close()
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
