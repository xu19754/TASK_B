"""Mink whole-body IK, free base + 12 legs + 6 Z1 joints and four fixed feet.
Geometry and mass centers are derived from the unmodified Task B URDF.
No MuJoCo dynamics steps; no Isaac base-pose writes after reset.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mink,mujoco
from scipy.spatial.transform import Rotation
from taskb_student.constants import JOINT_NAMES,LEG_Q0,ARM_Q0,JOINT_LOWER,JOINT_UPPER
from .kinematics import URDFKinematics,origin,wxyz,quat_matrix

class WholeBodyIK:
    def __init__(self,tcp_offset,effective_limits=None):
        self.fk=URDFKinematics();self.tcp_offset=np.array(tcp_offset)
        hard=np.array([JOINT_LOWER[:18],JOINT_UPPER[:18]]).T
        mid=hard.mean(axis=1);half=np.diff(hard,axis=1)[:,0]*.45
        self.limits_array=np.c_[mid-half,mid+half] if effective_limits is None else np.array(effective_limits)
        self.names=JOINT_NAMES[:18];self.feet=[leg+'_foot' for leg in ['FR','FL','RR','RL']]
        children={}
        for j in self.fk.joints: children.setdefault(j.find('parent').get('link'),[]).append(j)
        root=ET.Element('mujoco',model='TaskB_B2Z1_wholebody_kinematics')
        ET.SubElement(root,'compiler',angle='radian',balanceinertia='true')
        world=ET.SubElement(root,'worldbody')
        def add_link(parent,name,joint=None):
            attrs={'name':name}
            if joint is not None:
                t=origin(joint);attrs.update(pos=' '.join(map(str,t[:3,3])),quat=' '.join(map(str,wxyz(t[:3,:3]))))
            body=ET.SubElement(parent,'body',**attrs)
            if name=='base': ET.SubElement(body,'freejoint',name='floating_base')
            elif joint.get('name') in self.names:
                i=self.names.index(joint.get('name'))
                ET.SubElement(body,'joint',name=joint.get('name'),type='hinge',axis=joint.find('axis').get('xyz'),range=' '.join(map(str,self.limits_array[i])),limited='true')
            link=self.fk.tree.find(f"link[@name='{name}']")
            inertial=link.find('inertial')
            mass=max(float(inertial.find('mass').get('value')),1e-6) if inertial is not None else 1e-6
            com=origin(inertial)[:3,3] if inertial is not None else np.zeros(3)
            ET.SubElement(body,'inertial',mass=str(mass),pos=' '.join(map(str,com)),diaginertia='.01 .01 .01')
            if name in self.feet: ET.SubElement(body,'site',name=name+'_site',size='.005')
            if name=='link06': ET.SubElement(body,'site',name='attachment_site',pos=' '.join(map(str,self.tcp_offset)),size='.005')
            for child in children.get(name,[]): add_link(body,child.find('child').get('link'),child)
        add_link(world,'base')
        self.xml=ET.tostring(root,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.configuration=mink.Configuration(self.model)
        self.q_indices=[self.model.jnt_qposadr[self.model.joint(n).id] for n in self.names]
        self.v_indices=[self.model.jnt_dofadr[self.model.joint(n).id] for n in self.names]
        self.foot_tasks=[mink.FrameTask(n+'_site','site',position_cost=500.,orientation_cost=0.,lm_damping=.01) for n in self.feet]
        self.body_task=mink.FrameTask('base','body',position_cost=[80.,80.,80.],orientation_cost=[50.,50.,10.],lm_damping=.05)
        self.ee_task=mink.FrameTask('attachment_site','site',position_cost=30.,orientation_cost=5.,lm_damping=.05)
        self.posture=mink.PostureTask(self.model,cost=np.r_[np.zeros(6),np.ones(self.model.nv-6)*.002])
        self.com_task=mink.ComTask(cost=[10.,10.,0.])
        velocities={n:(.6 if i<12 else .5) for i,n in enumerate(self.names)}
        self.limits=[mink.ConfigurationLimit(self.model),mink.VelocityLimit(self.model,velocities)]
        self.last={};self.anchored=False
    def pack(self,base_pos,base_quat,joints):
        q=self.model.qpos0.copy();q[:3]=base_pos;q[3:7]=base_quat;q[self.q_indices]=np.asarray(joints)[:18]
        return q
    def anchor(self,base_pos,base_quat,joints):
        self.configuration.update(self.pack(base_pos,base_quat,joints))
        for task in self.foot_tasks: task.set_target_from_configuration(self.configuration)
        self.anchor_positions=np.array([self.configuration.get_transform_frame_to_world(n+'_site','site').translation() for n in self.feet])
        self.posture.set_target_from_configuration(self.configuration)
        self.com_task.set_target(self.configuration.data.subtree_com[0].copy())
        self.anchored=True
    def solve(self,base_pos,base_quat,joints,body_goal,body_quat,ee_goal=None,max_iters=12):
        if not self.anchored: self.anchor(base_pos,base_quat,joints)
        self.configuration.update(self.pack(base_pos,base_quat,joints))
        self.body_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(body_quat),np.asarray(body_goal)))
        tasks=[*self.foot_tasks,self.body_task,self.com_task,self.posture]
        constraints=[]
        if ee_goal is None:
            constraints=[mink.DofFreezingTask(self.model,dof_indices=self.v_indices[12:18])]
        else:
            self.ee_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(ee_goal[1]),np.asarray(ee_goal[0])))
            tasks.append(self.ee_task)
        okay=True;reason='finite_tracking_step'
        try:
            for i in range(max_iters):
                vel=mink.solve_ik(self.configuration,tasks,dt=.02,solver='daqp',limits=self.limits,constraints=constraints,damping=1e-5,safety_break=True)
                if not np.isfinite(vel).all(): raise RuntimeError('nonfinite velocity')
                self.configuration.integrate_inplace(vel,.02)
        except Exception as exc: okay=False;reason=type(exc).__name__+': '+str(exc)
        foot_error=max(float(np.linalg.norm(t.compute_error(self.configuration)[:3])) for t in self.foot_tasks)
        be=self.body_task.compute_error(self.configuration)
        ee=self.ee_task.compute_error(self.configuration) if ee_goal is not None else np.zeros(6)
        self.last={'success':okay and foot_error<.01,'reason':reason,'foot_error_m':foot_error,'body_position_error_m':float(np.linalg.norm(be[:3])),'body_angle_error_deg':float(np.rad2deg(np.linalg.norm(be[3:]))),'ee_position_error_m':float(np.linalg.norm(ee[:3])),'ee_angle_error_deg':float(np.rad2deg(np.linalg.norm(ee[3:]))),'planned_base':self.configuration.q[:7].tolist()}
        return self.configuration.q[self.q_indices].copy(),self.last.copy()
    def check_model(self):
        q=np.array(LEG_Q0+ARM_Q0);base=np.array([.02,-.03,.54]);quat=np.array([1.,0,0,0])
        self.configuration.update(self.pack(base,quat,q))
        frames=self.fk.frames(q);result=[]
        for name in [*self.feet,'link06']:
            mj=self.configuration.get_transform_frame_to_world(name,'body').as_matrix()
            expected=frames[name].copy();expected[:3,3]+=base
            error=float(np.linalg.norm(mj-expected))
            result.append({'body':name,'matrix_error':error})
        if max(r['matrix_error'] for r in result)>1e-9: raise RuntimeError('Full body FK mismatch')
        return result
    def support_torques(self,base,quat,joints,body_goal,body_quat,velocity_world,ang_velocity_world):
        """Stance contact-force allocation mapped by foot Jacobians to leg torques.
        Forces are only motor feedforward; no force is applied to the floating base.
        """
        data=mujoco.MjData(self.model);data.qpos[:]=self.pack(base,quat,joints)
        mujoco.mj_kinematics(self.model,data);mujoco.mj_comPos(self.model,data)
        feet=np.array([data.site_xpos[self.model.site(n+'_site').id] for n in self.feet])
        com=data.subtree_com[0].copy();mass=float(self.model.body_mass.sum())
        rel=feet[:,:2]-com[:2]
        orientation=Rotation.from_matrix(quat_matrix(body_quat)@quat_matrix(quat).T).as_rotvec()
        moment=np.array([120.,120.,40.])*orientation-np.array([15.,15.,8.])*np.asarray(ang_velocity_world)
        vertical=mass*(9.81+25.*(body_goal[2]-base[2])-8.*velocity_world[2])
        vertical=float(np.clip(vertical,.5*mass*9.81,1.5*mass*9.81))
        a=np.stack([np.ones(4),rel[:,1],-rel[:,0]])
        fz=a.T@np.linalg.solve(a@a.T+np.eye(3)*1e-8,np.array([vertical,moment[0],moment[1]]))
        fz=np.clip(fz,0.,mass*9.81*.6)
        fxy=mass*(20.*(np.asarray(body_goal)[:2]-np.asarray(base)[:2])-5.*np.asarray(velocity_world)[:2])
        fxy=np.clip(fxy,-.25*mass*9.81,.25*mass*9.81)
        forces=np.c_[fz[:,None]/max(fz.sum(),1.)*fxy[None,:],fz]
        torque=np.zeros(12)
        for k,n in enumerate(self.feet):
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacSite(self.model,data,jp,jr,self.model.site(n+'_site').id)
            torque-=jp[:,self.v_indices[:12]].T@forces[k]
        return np.clip(torque,-120.,120.),{'contact_force_targets':forces.tolist(),'model_mass':mass,'support_com':com.tolist(),'support_torque':torque.tolist()}
