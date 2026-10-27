import gymnasium as gym

from .constants import TASK_ID

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "taskb_student.env_cfg:TaskBEnvCfg"},
    )
