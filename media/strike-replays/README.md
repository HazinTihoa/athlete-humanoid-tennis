# Original strike dataset replays

Three forehand and three backhand clips from `data/djvkovic_npz_data_v5/`.
These are kinematic replays of GVHMR/GMR-derived G1 motions, not policy rollouts.
MP4 videos preserve the original 50 FPS at 1× speed; GIF previews are sampled
at 12.5 FPS. Each clip ends with a 0.4 s hold. Camera framing and the display
floor do not modify the source motion. `manifest.json` lists clip provenance
and checks the reconstructed positions against the recorded 34-body layout.

Regenerate from the repository root using the deployment Python environment:

```bash
uv pip install --python deploy/.venv/bin/python Pillow imageio-ffmpeg
MUJOCO_GL=egl deploy/.venv/bin/python scripts/render_strike_previews.py
```

Headless rendering requires a working EGL graphics driver. The script imports
the published joint-order constants and uses the repository's G1/racket model.
