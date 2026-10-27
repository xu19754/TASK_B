"""Experimental joint-path replay with physical grasp monitoring in the bench."""
import json
from pathlib import Path
import numpy as np
from experiments.mink_wholebody.controller import GraspController
from taskb_student.constants import CONTROL_DT

class BasketTransferController(GraspController):
    def __init__(self,config,ik,mode='grasp'):
        super().__init__(config,ik,mode)
        plan=json.loads((Path(__file__).parent/'experimental_joint_plan.json').read_text())
        if not plan['pass'] or plan['collision_checked'] or plan['object_clearance_checked']:
            raise ValueError('Unexpected experimental plan metadata')
        self.waypoints=[np.array(w['q'],float) for w in plan['waypoints']]
        self.grasp_confirmed=False;self.transfer_stages=['CARRY_PLAN','RELEASE','VERIFY_BASKET'];self.index=0;self.segment_start=None
    def change(self,state,reason=''):
        if self.state=='HOLD' and state=='DONE':
            return super().change('CARRY_PLAN' if self.grasp_confirmed else 'FAIL','grasp_confirmed' if self.grasp_confirmed else 'independent_grasp_check_failed')
        return super().change(state,reason)
    def step(self,p):
        if self.state not in self.transfer_stages: return super().step(p)
        self.time+=CONTROL_DT;self.age+=CONTROL_DT;self.active_time+=CONTROL_DT
        q=np.asarray(p['joint_pos']).reshape(19);base=np.asarray(p['root_position'])
        tilt=float(np.rad2deg(np.arccos(np.clip(-np.asarray(p['projected_gravity']).reshape(3)[2],-1,1))))
        if tilt>self.cfg['thresholds']['tilt_deg'] or np.linalg.norm(base[:2]-self.baseline[:2])>self.cfg['thresholds']['drift_m'] or p.get('illegal_contact',False):
            self.fail('carry_stability_limit');return self.target.copy()
        body=self.baseline.copy();body[2]=self.cfg['lift_body_height'];self.last_ik={'body_goal':body.tolist(),'joint_waypoint_index':self.index,'method':'experimental_precomputed_joint_path'}
        if self.state=='CARRY_PLAN':
            if self.segment_start is None:self.segment_start=self.target[:18].copy()
            waypoint=self.waypoints[self.index];duration=5.;u=min(self.age/duration,1.);u=u*u*(3.-2.*u)
            desired=(1.-u)*self.segment_start+u*waypoint
            self.target[:18]+=np.clip(desired-self.target[:18],-.3*CONTROL_DT,.3*CONTROL_DT)
            self.target[18]=0.
            error=float(np.max(np.abs(q[:18]-waypoint)));self.last_ik['waypoint_max_error_rad']=error
            if self.age>=duration and error<.06:
                self.index+=1;self.segment_start=self.target[:18].copy();self.age=0.
                if self.index==len(self.waypoints): self.change('RELEASE')
            elif self.age>duration+15.: self.fail('joint_waypoint_timeout')
        elif self.state=='RELEASE':
            self.target[18]=self.cfg['gripper_open']
            if abs(q[18]-self.target[18])<.04 and self.age>=6.:self.change('VERIFY_BASKET')
            elif self.age>12.:self.fail('release_timeout')
        elif self.state=='VERIFY_BASKET':
            self.target[18]=self.cfg['gripper_open']
            if self.age>=6.:self.change('DONE','basket_observation_complete')
        return self.target.copy()
