"""Keep the actual two-pad center fixed while closing, using only proprioception."""
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.mink_wholebody.controller import GraspController as BaseController

class GraspController(BaseController):
    def __init__(self,config,ik,mode='grasp'):
        super().__init__(config,ik,mode)
        self.nominal_rotation=self.ee_rotation.copy()
        self.fixed_goal_world=self.goal_world.copy()
        self.previous_tcp=None;self.approach_stall_s=0.;self.contact_latch=None;self.proprio_history=[]
    def step(self,p):
        if self.cfg.get('pinch_mode')=='fixed_tip':
            q=np.asarray(p['joint_pos']).reshape(19)
            frame=self.ik.fk.frames(q)['link06'];br=Rotation.from_quat(np.asarray(p['base_quat']).reshape(4),scalar_first=True).as_matrix()
            tcp=np.asarray(p['root_position']).reshape(3)+br@(frame[:3,3]+frame[:3,:3]@np.array(self.cfg['tcp_offset_link06']))
            speed=float(np.linalg.norm(tcp-self.previous_tcp)/.02) if self.previous_tcp is not None else float('inf')
            self.previous_tcp=tcp.copy()
            self.proprio_history.append((tcp.copy(),float(q[-1])))
            self.proprio_history=self.proprio_history[-21:]
            jaw_rate=float('inf')
            if len(self.proprio_history)==21:
                speed=float(np.linalg.norm(tcp-self.proprio_history[-11][0])/.2)
                jaw_rate=abs(float(q[-1])-self.proprio_history[0][1])/.4
            tolerance=float(self.cfg.get('proprioceptive_contact_latch_m',0.))
            if tolerance and self.state=='APPROACH' and self.age>=6.:
                blocked=q[-1]<-.08 and jaw_rate<.02
                close=np.linalg.norm(tcp-self.fixed_goal_world)<=tolerance
                self.approach_stall_s=self.approach_stall_s+.02 if blocked and close and speed<.002 else 0.
                if self.approach_stall_s>=.5:
                    self.fixed_goal_world=tcp.copy();self.nominal_rotation=br@frame[:3,:3];self.ee_rotation=self.nominal_rotation.copy()
                    self.contact_latch=dict(time=self.time,tcp=tcp.tolist(),gripper_angle=float(q[-1]),own_tcp_speed_m_s=speed)
                    self.change('CLOSE','proprioceptive_contact_stop')
            offset=np.array(self.cfg.get('pregrasp_offset_world',[0.,0.,0.]),float)
            fraction=1. if self.state=='PREGRASP' else max(0.,1.-(self.age+.02)/6.) if self.state=='APPROACH' else 0.
            self.goal_world=self.fixed_goal_world+fraction*offset
            result=super().step(p)
            if self.state=='APPROACH' and self.age>=float(self.cfg.get('early_close_after_s',1e9)):
                self.target[18]=0.;result[18]=0.
            if tolerance:self.last_ik['proprio_latch_diagnostics']=dict(tcp_speed_m_s=speed,gripper_angle_rate_rad_s=jaw_rate,stall_s=self.approach_stall_s,distance_m=float(np.linalg.norm(tcp-self.fixed_goal_world)))
            if self.contact_latch:self.last_ik['contact_pose_latch']=self.contact_latch
            return result
        q=np.asarray(p['joint_pos']).reshape(19)
        center,alpha,moving=self.ik.update_pinch_frame(q)
        self.cfg['tcp_offset_link06']=center.tolist()
        beta=0.
        if self.state in ['LIFT','HOLD','DONE']:
            fraction=min(self.age/float(self.cfg.get('lift_tilt_duration_s',3.)),1.) if self.state=='LIFT' else 1.
            beta=np.deg2rad(float(self.cfg.get('lift_tilt_deg',0.)))*fraction
        self.ee_rotation=self.nominal_rotation@Rotation.from_euler('y',alpha-beta).as_matrix()
        result=super().step(p)
        self.last_ik.update(actual_gripper_angle=float(q[-1]),pinch_center_link06=center.tolist(),pad_leveling_angle_rad=alpha,fixed_pad_link06=self.ik.fixed_pad.tolist(),moving_pad_link06=moving.tolist())
        return result
