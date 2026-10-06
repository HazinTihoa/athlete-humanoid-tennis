"""Checkpoint the landing curriculum for tasks that explicitly select this runner."""

from .runner import MotionTrackingOnPolicyRunner


class LandingCurriculumOnPolicyRunner(MotionTrackingOnPolicyRunner):
  def save(self, path: str, infos=None):
    command = self.env.unwrapped.command_manager.get_term("motion")
    if command.cfg.landing_target_std_final is not None:
      infos = dict(infos or {})
      infos["landing_target_curriculum_steps"] = (
        self.env.unwrapped.common_step_counter
        + command.landing_target_curriculum_step_offset
      )
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    command = self.env.unwrapped.command_manager.get_term("motion")
    if command.cfg.landing_target_std_final is not None:
      restore_progress = load_cfg is None or load_cfg.get("iteration", False)
      steps = (infos or {}).get("landing_target_curriculum_steps", 0) if restore_progress else 0
      command.landing_target_curriculum_step_offset = (
        steps - self.env.unwrapped.common_step_counter
      )
      command.update_landing_target_std_curriculum()
    return infos
