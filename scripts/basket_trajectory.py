"""Common-clock C2 joint trajectories with sampled bound/speed checks."""
import numpy as np
from scipy.interpolate import CubicSpline


class JointTrajectory:
    def __init__(self, nodes, lower, upper, velocity, peak=.04, parameters=None):
        q = np.asarray(nodes, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(lower) or len(q) < 2 or not np.isfinite(q).all():
            raise ValueError('至少需要两个有效的七关节姿态')
        if parameters is None:
            keep = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-7]
            q = q[keep]
            if len(q) < 2:
                raise ValueError('记录的姿态相同，没有可回放的路径')
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            parameters = distance / distance[-1]
        self.nodes = q
        self.parameters = np.asarray(parameters)
        self.spline = CubicSpline(self.parameters, q, bc_type='clamped')
        sample = np.linspace(0, 1, 1001)
        position = self.spline(sample)
        if np.any(position < np.asarray(lower) - 1e-9) or np.any(position > np.asarray(upper) + 1e-9):
            raise ValueError('平滑插值将超出关节限位；请补充中间姿态')
        speed = np.minimum(np.asarray(velocity) * .5, peak)
        acceleration = np.r_[[.10] * (len(lower) - 1), .05]
        self.duration = max(2., float(np.max(np.abs(self.spline(sample, 1)) / speed)),
                            float(np.sqrt(np.max(np.abs(self.spline(sample, 2)) / acceleration))))
        self.peak_velocity = np.max(np.abs(self.spline(sample, 1)), axis=0) / self.duration
        self.peak_acceleration = np.max(np.abs(self.spline(sample, 2)), axis=0) / self.duration ** 2

    def at(self, elapsed):
        return self.spline(np.clip(elapsed / self.duration, 0, 1))

    def details(self):
        return dict(q_nodes_rad=self.nodes.tolist(), parameters=self.parameters.tolist(), duration_s=self.duration,
                    peak_target_velocity_rad_s=self.peak_velocity.tolist(),
                    peak_target_acceleration_rad_s2=self.peak_acceleration.tolist(),
                    interpolation='C2 cubic spline, clamped zero endpoint velocity, common joint clock')
