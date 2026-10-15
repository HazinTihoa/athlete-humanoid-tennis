# GPU3-2 Full-flight Student ONNX

- Model: `model_45000.onnx`, exported from iteration 45,000 of the GPU3-2 run.
- Training run: `gpu3-2_gpu4-6base_model38000_footforce20_torsoforce20_returnhomespeed_timeout20_originalobs_fail05_8192env_add30000iter`.
- Input: 631 observation values; output: 29 joint actions; control: 50 Hz.
- Landing target: startup-frame XY `(5, 0)` m, observed in the **current root frame**.
- Racket velocity observation: instantaneous velocity **relative to the pelvis**, without EMA.
- `params/` contains the saved training configurations as provenance, not launch configuration files. Their original machine paths are not used by the Quick Demo.
- `manifest.json` records file checksums and the source checkpoint hash. The source `model_45000.pt` is not included in this directory; ONNX inference does not need it.

Run the Quick Demo from the repository root using `uv run python scripts/demo_mujoco.py` after the environment setup in the main README. No reference-motion dataset is required.
