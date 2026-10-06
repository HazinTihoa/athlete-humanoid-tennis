# MIT License

Copyright (c) 2026 Anonymous Authors (ATHLETE)

Portions of this software are derived from TaskNPoint
(https://github.com/wernerb43/tasknpoint), which is distributed under the
following license:

> Copyright (c) 2026 Blake Werner, Ilona Demler, Pietro Perona, Aaron D. Ames

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Third-party components

The MIT license above applies to the original code in this repository. The
following components retain their own licenses, which take precedence for the
respective files:

- `athlete/src/athlete/goal_cond_tracking/rl/tppo.py` (and its rollout storage)
  adapt the TPPO algorithm from Instinct-RL
  (https://github.com/project-instinct/instinct_rl), licensed under
  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). These files
  may not be used for commercial purposes.
- Unitree G1 robot model and meshes (`robots/`, `deploy/policies/*/robot/`):
  BSD 3-Clause, Unitree Robotics (see `deploy/policies/m14_11/robot/LICENSE.Unitree`).
- The deployment stack in `deploy/` is based on
  https://github.com/sesteban951/deploy_robot; see that project for its license.
- Locomotion references are derived from the Ubisoft La Forge Animation
  Dataset (LAFAN1), licensed under CC BY-NC-ND 4.0.
- External Python dependencies (mjlab, RSL-RL, MuJoCo, PyTorch, ONNX Runtime,
  and others installed by `uv sync`) are governed by their own licenses.
