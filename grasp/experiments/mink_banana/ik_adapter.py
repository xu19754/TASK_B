"""Actual gripper-angle dependent pinch frame; MuJoCo remains kinematics-only."""
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
from taskb_student.constants import LEG_Q0,ARM_Q0

class BananaIK(WholeBodyIK):
    def __init__(self,tcp_offset,effective_limits=None):
        super().__init__(tcp_offset,effective_limits)
        q=np.array(LEG_Q0+ARM_Q0,float);q[-1]=0.
        f=self.fk.frames(q);self.fixed_pad=np.array([.230,0.,-.009])
        closed=np.linalg.inv(f['link06'])@f['gripperMover']
        self.moving_pad_local=np.linalg.inv(closed)@np.r_[self.fixed_pad,1.]
        self.site_id=self.model.site('attachment_site').id
    def pinch_geometry(self,joints):
        q=np.asarray(joints,float).copy();q[-1]=np.clip(q[-1],-1.57,0.)
        f=self.fk.frames(q);moving=(np.linalg.inv(f['link06'])@f['gripperMover']@self.moving_pad_local)[:3]
        center=(self.fixed_pad+moving)*.5
        # Compute the limiting direction as the jaw closes, avoiding atan2(0,0).
        q_angle=q.copy();q_angle[-1]=min(q[-1],-1e-4)
        fa=self.fk.frames(q_angle);ma=(np.linalg.inv(fa['link06'])@fa['gripperMover']@self.moving_pad_local)[:3]
        delta=ma-self.fixed_pad
        alpha=float(np.arctan2(-delta[0],delta[2]))
        return center,alpha,moving
    def update_pinch_frame(self,joints):
        center,alpha,moving=self.pinch_geometry(joints)
        self.tcp_offset=center.copy();self.model.site_pos[self.site_id]=center
        return center,alpha,moving
