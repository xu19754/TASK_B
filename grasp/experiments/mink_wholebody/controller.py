"""Mink whole-body stance, crouch, approach and physical grasp state machine."""
import numpy as np
from scipy.spatial.transform import Rotation
from taskb_student.constants import ARM_Q0,LEG_Q0,CONTROL_DT
from .kinematics import quat_matrix,wxyz,transform

class GraspController:
    def __init__(self,config,ik,mode='grasp'):
        self.cfg=config;self.ik=ik;self.mode=mode
        self.state='SETTLE';self.age=0.;self.time=0.;self.reason=''
        self.target=np.array(LEG_Q0+ARM_Q0,float);self.history=[];self.baseline=None
        self.last_ik={};self.active=False;self.active_time=0.;self.failures=0
        self.goal_world=np.array(config['grasp_world'],float)
        self.ee_rotation=Rotation.from_euler('xyz',config['grasp_rpy']).as_matrix()
        self.last_goal=None
    def change(self,state,reason=''):
        self.history.append({'time':self.time,'from':self.state,'to':state,'reason':reason})
        self.state=state;self.age=0.;self.reason=reason
    def fail(self,reason): self.change('FAIL',reason)
    def step(self,p):
        self.time+=CONTROL_DT;self.age+=CONTROL_DT
        q=np.asarray(p['joint_pos']).reshape(19);base=np.asarray(p['root_position'])
        quat=np.asarray(p['base_quat']).reshape(4);rot=quat_matrix(quat)
        tilt=float(np.rad2deg(np.arccos(np.clip(-np.asarray(p['projected_gravity']).reshape(3)[2],-1,1))))
        if self.state=='SETTLE':
            if self.age>=6.:
                self.baseline=base.copy();self.start_height=float(base[2])
                yaw=Rotation.from_matrix(rot).as_euler('xyz')[2]
                self.body_quat=wxyz(Rotation.from_euler('xyz',[0.,self.cfg['body_pitch_rad'],yaw]).as_matrix())
                self.ik.anchor(base,quat,q);self.target=q.copy();self.target[12:]=ARM_Q0
                self.active=True;self.change('SQUAT')
            return self.target.copy()
        if self.state in ['DONE','FAIL']: return self.target.copy()
        self.active_time+=CONTROL_DT
        if tilt>self.cfg['thresholds']['tilt_deg']: self.fail('tilt_limit');return self.target.copy()
        if np.linalg.norm(base[:2]-self.baseline[:2])>self.cfg['thresholds']['drift_m']: self.fail('drift_limit');return self.target.copy()
        if p.get('illegal_contact',False): self.fail('illegal_body_contact');return self.target.copy()
        body_goal=self.baseline.copy();body_goal[2]=self.cfg['body_height']
        if self.state=='SQUAT':
            u=min(self.age/self.cfg['squat_duration_s'],1.);u=u*u*(3.-2.*u)
            body_goal[2]=(1.-u)*self.start_height+u*self.cfg['body_height']
        if self.state=='LIFT':
            u=min(self.age/6.,1.);body_goal[2]+=(self.cfg['lift_body_height']-self.cfg['body_height'])*u
        if self.state=='HOLD': body_goal[2]=self.cfg['lift_body_height']
        ee=None
        if self.state in ['PREGRASP','APPROACH','CLOSE','LIFT','HOLD']:
            goal=self.goal_world.copy()
            rotation_goal=self.ee_rotation
            if self.state=='PREGRASP':
                goal[2]+=self.cfg['pregrasp_up']
                if self.age<=.021:
                    own_pose=self.ik.fk.frames(q)['link06']@transform(self.cfg['tcp_offset_link06'])
                    self.pre_start_pos=base+rot@own_pose[:3,3]
                    self.pre_start_rot=rot@own_pose[:3,:3]
                u=min(self.age/6.,1.);u=u*u*(3.-2.*u)
                goal=(1.-u)*self.pre_start_pos+u*goal
                rv=Rotation.from_matrix(self.pre_start_rot.T@self.ee_rotation).as_rotvec()
                rotation_goal=self.pre_start_rot@Rotation.from_rotvec(u*rv).as_matrix()
            if self.state=='APPROACH': goal[2]+=self.cfg['pregrasp_up']*max(0.,1.-self.age/6.)
            if self.state=='LIFT': goal[2]+=self.cfg['lift_up']*min(1.,self.age/6.)
            if self.state=='HOLD': goal[2]+=self.cfg['lift_up']
            self.last_goal=goal.copy();ee=(goal,wxyz(rotation_goal))
        candidate,report=self.ik.solve(base,quat,q,body_goal,self.body_quat,ee,max_iters=12)
        self.last_ik=report
        own_frames=self.ik.fk.frames(q)
        feet_actual=np.array([base+rot@own_frames[n][:3,3] for n in self.ik.feet])
        self.last_ik['actual_foot_anchor_error_m']=float(np.linalg.norm(feet_actual-self.ik.anchor_positions,axis=1).max())
        if self.active_time>3. and self.last_ik['actual_foot_anchor_error_m']>.03:
            self.fail('foot_anchor_drift');return self.target.copy()
        if not report['success']:
            self.failures+=1
            if self.failures>=20: self.fail('wholebody_ik_failure')
            return self.target.copy()
        self.failures=0
        self.target[:12]+=np.clip(candidate[:12]-self.target[:12],-.6*CONTROL_DT,.6*CONTROL_DT)
        if ee is not None:
            self.target[12:18]+=np.clip(candidate[12:18]-self.target[12:18],-.4*CONTROL_DT,.4*CONTROL_DT)
            actual=self.ik.fk.frames(q)['link06']@transform(self.cfg['tcp_offset_link06'])
            actual_pos=base+rot@actual[:3,3]
            pe=float(np.linalg.norm(actual_pos-ee[0]))
            oe=float(np.rad2deg(Rotation.from_matrix((rot@actual[:3,:3]).T@quat_matrix(ee[1])).magnitude()))
            self.last_ik.update(actual_position_error=pe,actual_orientation_error_deg=oe,body_goal=body_goal.tolist())
            arrived=pe<=.005 and oe<=5.
        else: arrived=False
        if self.state=='SQUAT' and self.age>=self.cfg['squat_duration_s']+2.:
            if abs(base[2]-self.cfg['body_height'])<.04:
                self.ik.anchor(base,quat,q)
                self.baseline[:2]=base[:2]
                self.change('POSTURE_HOLD' if self.mode=='posture' else 'OPEN')
        elif self.state=='POSTURE_HOLD' and self.age>=3.: self.change('DONE','posture_only')
        elif self.state=='OPEN':
            self.target[18]=self.cfg['gripper_open']
            if abs(q[18]-self.target[18])<.04 and self.age>1.: self.change('PREGRASP')
        elif self.state=='PREGRASP' and arrived and self.age>=6.: self.change('APPROACH')
        elif self.state=='APPROACH' and arrived and self.age>=6.: self.change('DONE' if self.mode=='reach' else 'CLOSE')
        elif self.state=='CLOSE':
            self.target[18]=0.
            if self.age>=8.: self.change('LIFT')
        elif self.state=='LIFT' and arrived and self.age>=6.: self.change('HOLD')
        elif self.state=='HOLD' and self.age>=4.: self.change('DONE')
        limit=self.cfg['stage_limits_s'].get(self.state,100.)
        if self.age>limit: self.fail(self.state.lower()+'_timeout')
        return self.target.copy()
