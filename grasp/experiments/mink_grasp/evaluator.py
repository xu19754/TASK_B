"""Privileged measurements isolated from the grasp controller."""
import numpy as np
from .mink_adapter import quat_matrix

class Evaluator:
    def __init__(self,config,vertices):
        self.cfg=config;self.vertices=vertices
        self.initial_z=None;self.hold=0.;self.max_rise=0.;self.max_hold=0.;self.baseline=None
        self.last={};self.rows=[]
    def observe(self,t,state,obj_position,obj_quat,wrist_position,wrist_quat,gripper,base_position,tilt,illegal_contact,jaw_forces):
        if self.initial_z is None or state=='SETTLE':
            self.initial_z=float(obj_position[2]);self.baseline=np.array(base_position)
        rot=quat_matrix(obj_quat)
        world=self.vertices@rot.T+obj_position
        bottom=float(world[:,2].min());rise=float(obj_position[2]-self.initial_z)
        grip_point=np.array(obj_position)+rot@np.array(self.cfg.get('grasp_local_point',[0.,-.070,0.]))
        gp=quat_matrix(wrist_quat).T@(grip_point-wrist_position)
        upper=-np.sin(float(gripper))*.11+.015
        in_jaws=bool(.11<=gp[0]<=.25 and abs(gp[1])<=.065 and -.055<=gp[2]<=upper+.025)
        drift=float(np.linalg.norm(np.asarray(base_position)[:2]-self.baseline[:2]))
        th=self.cfg['thresholds']
        elevated=rise>=th['rise_m'] and bottom>=th['bottom_m']
        stable=tilt<=th['tilt_deg'] and drift<=th['drift_m'] and not illegal_contact
        dual_contact=bool(min(jaw_forces)>.25)
        valid=bool(elevated and stable and in_jaws and dual_contact)
        self.hold=self.hold+.02 if valid else 0.
        self.max_rise=max(self.max_rise,rise);self.max_hold=max(self.max_hold,self.hold)
        self.last=dict(sim_time=t,state=state,initial_root_z=self.initial_z,object_root=list(map(float,obj_position)),rise_m=rise,bottom_m=bottom,grip_point_in_wrist=gp.tolist(),in_jaws=in_jaws,jaw_forces=list(map(float,jaw_forces)),dual_contact=dual_contact,tilt_deg=tilt,drift_m=drift,illegal_contact=bool(illegal_contact),valid_hold_s=self.hold)
        return self.last
    def result(self):
        return {'success':self.max_hold>=self.cfg['thresholds']['hold_s'],'max_rise_m':self.max_rise,'max_valid_hold_s':self.max_hold,'last':self.last}
