from taskb_student.constants import ARM_Q0
from taskb_student.low_level import LowLevelController


class AlgSolution:
    def __init__(self, command=(0.0, 0.0, 0.0)):
        self.controller = LowLevelController()
        self.command = command

    def reset(self):
        self.controller.reset()

    def select_command(self, obs, current_score):
        """Implement perception/navigation here. Commands are (vx, 0, yaw_rate)."""
        return self.command

    def select_arm_target(self, obs, current_score):
        """Return 6 Z1 joint angles + 1 gripper angle, all in radians."""
        return ARM_Q0

    def predicts(self, obs, current_score):
        action = self.controller.step(
            obs["proprio"], self.select_command(obs, current_score),
            self.select_arm_target(obs, current_score),
        )
        return {"action": action.tolist(), "giveup": False}
