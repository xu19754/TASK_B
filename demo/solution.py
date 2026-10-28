from pathlib import Path
import uuid

from taskb_student.constants import ARM_Q0
from taskb_student.localization import Localization
from taskb_student.low_level import LowLevelController
from taskb_student.motion import Motion
from taskb_student.object_finder import ObjectFinder
from taskb_student.traversal import Traversal


class AlgSolution:
    def __init__(self, command=(0.0, 0.0, 0.0), target_position=None, *,
                 scan_before_start=True, detector=None, scan_output_dir="runs/object_scan", scan_use_wrist=True):
        self.controller = LowLevelController()
        self.command = command
        self.localization = Localization()
        self.motion = Motion(self.localization)
        self.target_position = target_position
        self.traversal = None
        self.scan_use_wrist = scan_use_wrist
        self.object_finder = ObjectFinder(self.localization, detector=detector, use_wrist_camera=scan_use_wrist) if scan_before_start else None
        self.scan_output_dir = Path(scan_output_dir) if scan_output_dir is not None else None
        self.scan_result_path = None
        self._scan_reported = False
        self._previous_command = None

    def reset(self):
        self.controller.reset()
        self.localization.reset()
        if self.traversal is not None:
            self.traversal.reset()
        if self.object_finder is not None:
            self.object_finder.reset()
        self._scan_reported = False
        self.scan_result_path = None
        self._previous_command = None

    def select_command(self, obs, current_score):
        """Prioritize pickup traversal, then a single target, then the manual command."""
        if self.traversal is not None:
            return self.traversal.compute_command()
        if self.target_position is not None:
            return self.motion.compute_command(self.target_position)
        return self.command

    def set_traversal(self, target_poses, obstacle_positions, **options):
        """Start traversal. pickup_enabled=False visits each pose for dwell_seconds
        (default 3), retains obstacles and keeps the arm folded. Otherwise an
        external grasp controller must call self.traversal.confirm_pickup().
        """
        self.traversal = Traversal(self.localization, target_poses, obstacle_positions, **options)

    def restart_scan(self):
        """Rescan from the current pose between control steps without resetting localization."""
        if self.object_finder is None:
            self.object_finder = ObjectFinder(self.localization, use_wrist_camera=self.scan_use_wrist)
        else:
            self.object_finder.reset()
        self._scan_reported = False
        self.scan_result_path = None

    def select_arm_target(self, obs, current_score):
        """Return 6 Z1 joint angles + 1 gripper angle, all in radians."""
        return ARM_Q0

    def predicts(self, obs, current_score):
        # The previous command ran for one simulation step before this observation.
        if self._previous_command is not None:
            self.localization.update(self._previous_command)
            self._previous_command = None
        scanning = self.object_finder is not None and not self.object_finder.ready
        if scanning:
            command = self.object_finder.compute_command(obs)
            if self.object_finder.done and not self._scan_reported:
                if self.object_finder.observed_yaw is not None:
                    self.localization.correct_heading(self.object_finder.observed_yaw)
                if self.scan_output_dir is not None:
                    directory = self.scan_output_dir / f"scan-{uuid.uuid4().hex[:8]}"
                    self.scan_result_path = self.object_finder.save_results(directory)
                report = self.object_finder.report()
                print(f"[ObjectFinder] {report['status']}: {report['confirmed_count']}/18; "
                      f"reason={report['reason']}; report={self.scan_result_path}", flush=True)
                self._scan_reported = True
        else:
            command = self.select_command(obs, current_score)
        command = self.localization.prepare_command(command)
        if scanning:
            arm_target = self.object_finder.arm_target
        elif self.traversal is not None and not self.traversal.pickup_enabled:
            arm_target = ARM_Q0
        else:
            arm_target = self.select_arm_target(obs, current_score)
        action = self.controller.step(
            obs["proprio"], command, arm_target,
        )
        self._previous_command = command.copy()
        return {"action": action.tolist(), "giveup": False}
