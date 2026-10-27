"""Step-driven diagnostic controller. Never receives object simulation state."""
import numpy as np
from scipy.spatial.transform import Rotation
from taskb_student.constants import ARM_Q0,CONTROL_DT
from .mink_adapter import wxyz,quat_matrix

class GraspController:
    def __init__(self,config,ik,mode='grasp'):
        self.cfg=config;self.ik=ik;self.mode=mode
        self.state='SETTLE';self.age=0.;self.time=0.;self.reason=''
        self.target=np.array(ARM_Q0,float);self.target[6]=0.
        self.last_ik={};self.history=[];self.baseline=None;self.ik_failures=0
        self.goal_world=np.array(config['grasp_world'],float)
        self.rotation_world=Rotation.from_euler('xyz',config['grasp_rpy']).as_matrix()
        self.last_goal=None
    def change(self,state,reason=''):
        self.history.append({'time':self.time,'from':self.state,'to':state,'reason':reason})
        self.state=state;self.age=0.;self.reason=reason
    def fail(self,reason): self.change('FAIL',reason)
    def step(self,proprio):
        self.time+=CONTROL_DT;self.age+=CONTROL_DT
        q=np.asarray(proprio['joint_pos']).reshape(19)
        base=np.asarray(proprio['root_position']).reshape(3)
        rot=quat_matrix(np.asarray(proprio['base_quat']).reshape(4))
        tilt=float(np.rad2deg(np.arccos(np.clip(-np.asarray(proprio['projected_gravity']).reshape(3)[2],-1,1))))
        if not np.isfinite(q).all(): self.fail('nonfinite_state')
        if self.state=='SETTLE':
            self.target[:6]=ARM_Q0[:6]
            if self.age>=self.cfg['stage_limits_s']['SETTLE']:
                self.baseline=base.copy();self.change('OPEN')
        elif self.state not in ['DONE','FAIL']:
            drift=float(np.linalg.norm(base[:2]-self.baseline[:2]))
            if tilt>self.cfg['thresholds']['tilt_deg']: self.fail('tilt_limit')
            elif drift>self.cfg['thresholds']['drift_m']: self.fail('drift_limit')
            elif proprio.get('illegal_contact',False): self.fail('illegal_body_contact')
        if self.state=='OPEN':
            self.target[6]=self.cfg['gripper_open']
            if abs(q[18]-self.cfg['gripper_open'])<.04 and self.age>1.:
                self.change('PREGRASP')
            elif self.age>self.cfg['stage_limits_s']['OPEN']: self.fail('open_timeout')
        elif self.state in ['PREGRASP','APPROACH','LIFT']:
            goal=self.goal_world.copy()
            if self.state=='PREGRASP': goal[2]+=self.cfg['pregrasp_up']
            if self.state=='APPROACH': goal[2]+=self.cfg['pregrasp_up']*max(0.,1.-self.age/6.)
            if self.state=='LIFT': goal[2]+=self.cfg['lift_up']*min(1.,self.age/6.)
            local=rot.T@(goal-base);quat=wxyz(rot.T@self.rotation_world)
            solution,report=self.ik.solve(local,quat,q[12:18],max_iters=30)
            self.last_ik=report;self.last_goal=goal.copy()
            if not report['success']:
                self.ik_failures+=1
                if self.ik_failures>=10: self.fail(self.state.lower()+'_ik_unreachable')
            else:
                self.ik_failures=0
                # Check the entire first route and all later short commands.
                check=self.ik.path_check(q[12:18],solution,float(base[2]),float(q[18])) if self.age<=.021 else None
                if check is not None and not check['safe']:
                    self.last_ik['path_check']=check;self.fail('unsafe_'+self.state.lower()+'_path')
                else:
                    delta=np.clip(solution-self.target[:6],-self.cfg['arm_speed_rad_s']*CONTROL_DT,self.cfg['arm_speed_rad_s']*CONTROL_DT)
                    self.target[:6]+=delta
                    actual=self.ik.fk.tcp(q[12:18],self.cfg['tcp_offset_link06'])
                    pe=float(np.linalg.norm(actual[:3,3]-local))
                    oe=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3].T@quat_matrix(quat)).magnitude()))
                    self.last_ik.update(actual_position_error=pe,actual_orientation_error_deg=oe)
                    if pe<=self.cfg['thresholds']['ik_position_m'] and oe<=self.cfg['thresholds']['ik_orientation_deg'] and self.age>(6. if self.state in ['APPROACH','LIFT'] else .5):
                        if self.state=='PREGRASP': self.change('APPROACH')
                        elif self.state=='APPROACH': self.change('DONE' if self.mode=='reach' else 'CLOSE')
                        elif self.state=='LIFT': self.change('HOLD')
            if self.state not in ['DONE','FAIL'] and self.age>self.cfg['stage_limits_s'][self.state]: self.fail(self.state.lower()+'_timeout')
        elif self.state=='CLOSE':
            self.target[6]=0.
            # Contact may prevent reaching zero. Wait long enough for limited closing.
            if self.age>=7.: self.change('LIFT')
            elif self.age>self.cfg['stage_limits_s']['CLOSE']: self.fail('close_timeout')
        elif self.state=='HOLD':
            self.target[6]=0.
            if self.age>=self.cfg['stage_limits_s']['HOLD']: self.change('DONE')
        return self.target.copy()

