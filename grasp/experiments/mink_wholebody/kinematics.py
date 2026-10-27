"""Z1 Mink adapter, following BinWalker IKMink's FrameTask/PostureTask/DAQP pattern.
Source: iit-DLSLab/trash-collection-isaaclab, commit
76c51c4c08e5762d00c75f36cb07aa3e525d6990/deploy/ik_mink.py (BSD-3-Clause).
Original source and license retained in reference/. Derived geometry comes only
from this assignment's unmodified URDF. MuJoCo is used for kinematics, never physics.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco,mink
from taskb_student.constants import URDF_PATH,ARM_Q0,JOINT_NAMES,LEG_Q0,JOINT_LOWER,JOINT_UPPER

def transform(xyz=(0,0,0),rpy=(0,0,0)):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz
    return t
def origin(j):
    o=j.find('origin')
    return transform(np.fromstring(o.get('xyz','0 0 0'),sep=' '),np.fromstring(o.get('rpy','0 0 0'),sep=' ')) if o is not None else np.eye(4)
def wxyz(matrix):
    return np.roll(Rotation.from_matrix(matrix).as_quat(),1)
def quat_matrix(quat):
    return Rotation.from_quat(np.roll(quat,-1)).as_matrix()

class URDFKinematics:
    """Independent FK used for model cross-checking and conservative path checks."""
    def __init__(self):
        self.tree=ET.parse(URDF_PATH).getroot()
        self.joints=self.tree.findall('joint')
    def frames(self,q):
        values=dict(zip(JOINT_NAMES,np.r_[LEG_Q0,np.asarray(q)])) if len(q)==7 else dict(zip(JOINT_NAMES,q))
        result={'base':np.eye(4)}
        pending=list(self.joints)
        while pending:
            remaining=[]
            for j in pending:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in result:
                    remaining.append(j);continue
                motion=np.eye(4)
                if j.get('type')!='fixed':
                    axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                    motion[:3,:3]=Rotation.from_rotvec(axis*values.get(j.get('name'),0.)).as_matrix()
                result[child]=result[parent]@origin(j)@motion
            if len(remaining)==len(pending): raise RuntimeError('Unresolved URDF tree')
            pending=remaining
        return result
    def tcp(self,q,offset):
        f=self.frames(np.r_[q,0.] if len(q)==6 else q)['link06']
        return f@transform(offset)

class MinkAdapter:
    def __init__(self,tcp_offset=(.21,0.,.02),effective_limits=None):
        self.tcp_offset=np.array(tcp_offset,float)
        self.fk=URDFKinematics()
        hard=np.array([JOINT_LOWER[12:18],JOINT_UPPER[12:18]]).T
        mid=hard.mean(axis=1);half=(hard[:,1]-hard[:,0])*.45
        self.limits_array=np.c_[mid-half,mid+half] if effective_limits is None else np.array(effective_limits,float)
        root=ET.Element('mujoco',model='TaskB_Z1_URDF_kinematics')
        ET.SubElement(root,'compiler',angle='radian')
        world=ET.SubElement(root,'worldbody')
        mt=origin(self.fk.tree.find("joint[@name='base_static_joint']"))
        parent=ET.SubElement(world,'body',name='link00',pos=' '.join(map(str,mt[:3,3])),quat=' '.join(map(str,wxyz(mt[:3,:3]))))
        for i in range(1,7):
            j=self.fk.tree.find(f"joint[@name='joint{i}']")
            t=origin(j)
            body=ET.SubElement(parent,'body',name=f'link0{i}',pos=' '.join(map(str,t[:3,3])),quat=' '.join(map(str,wxyz(t[:3,:3]))))
            ET.SubElement(body,'inertial',pos='0 0 0',mass='1',diaginertia='.01 .01 .01')
            ET.SubElement(body,'joint',name=f'joint{i}',type='hinge',axis=j.find('axis').get('xyz'),limited='true',range=' '.join(map(str,self.limits_array[i-1])))
            parent=body
        ET.SubElement(parent,'site',name='attachment_site',pos=' '.join(map(str,self.tcp_offset)),size='.005')
        self.xml=ET.tostring(root,encoding='unicode')
        (Path(__file__).parent/'models/z1_urdf_reduced.xml').write_text(self.xml)
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.configuration=mink.Configuration(self.model)
        self.end_effector_task=mink.FrameTask('attachment_site','site',position_cost=1.,orientation_cost=1.,lm_damping=.1)
        self.posture_task=mink.PostureTask(self.model,cost=1e-6)
        self.tasks=[self.end_effector_task,self.posture_task]
        self.limits=[mink.ConfigurationLimit(self.model)]
        self.last={}
    def pose(self,q):
        self.configuration.update(np.asarray(q,float))
        return self.configuration.get_transform_frame_to_world('attachment_site','site').as_matrix()
    def solve(self,target_pos,target_quat,q_actual,max_iters=120):
        q=np.asarray(q_actual,float).copy()
        outside=np.maximum(self.limits_array[:,0]-q,q-self.limits_array[:,1])
        if np.any(outside>1e-6):
            self.last={'success':False,'reason':'actual_outside_soft_limits','iterations':0,'actual_q':q.tolist(),'max_limit_excess_rad':float(outside.max())}
            return q,self.last.copy()
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)
        goal=mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(target_quat,float)),np.asarray(target_pos,float))
        self.end_effector_task.set_target(goal)
        initial=float(np.linalg.norm(self.end_effector_task.compute_error(self.configuration)[:3]))
        success=False;reason='iteration_limit'
        for i in range(max_iters):
            try:
                vel=mink.solve_ik(self.configuration,self.tasks,dt=.02,solver='daqp',damping=1e-5,limits=self.limits,safety_break=True)
                if not np.isfinite(vel).all(): reason='nonfinite';break
                self.configuration.integrate_inplace(vel,.02)
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc);break
            err=self.end_effector_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3])<=.003 and np.linalg.norm(err[3:])<=np.deg2rad(3.):
                success=True;reason='converged';break
        err=self.end_effector_task.compute_error(self.configuration)
        self.last={'success':success,'reason':reason,'iterations':i+1,'initial_position_error':initial,'position_error':float(np.linalg.norm(err[:3])),'orientation_error_deg':float(np.rad2deg(np.linalg.norm(err[3:])))}
        return self.configuration.q.copy(),self.last.copy()
    def check_model(self):
        tests=[ARM_Q0[:6],[.25,1.7,-1.2,-.5,.3,1.2],[-.3,2.1,-2.,.4,-.3,-.6]]
        result=[]
        for q in tests:
            a=self.pose(q);b=self.fk.tcp(q,self.tcp_offset)
            result.append({'q':list(q),'position_error':float(np.linalg.norm(a[:3,3]-b[:3,3])),'rotation_error':float(np.linalg.norm(a[:3,:3]-b[:3,:3]))})
        if max(x['position_error']+x['rotation_error'] for x in result)>1e-9: raise RuntimeError('URDF/MuJoCo FK mismatch')
        return result
    def path_check(self,q_start,q_end,base_height=.6,gripper=-1.3):
        """Sampled centerlines/finger envelope vs trunk, leg capsules and floor.
        Conservative diagnostic screening, not a general collision planner."""
        failures=[];minimum_floor=10.
        legs=self.fk.frames(np.r_[LEG_Q0,ARM_Q0])
        for alpha in np.linspace(0,1,51):
            q=(1-alpha)*np.asarray(q_start)+alpha*np.asarray(q_end)
            f=self.fk.frames(np.r_[q,gripper])
            points=[]
            for a,b in [('link02','link03'),('link03','link04'),('link04','link06')]:
                points.extend(np.linspace(f[a][:3,3],f[b][:3,3],25))
            wrist=f['link06']
            points.extend([(wrist@np.r_[x,y,z,1])[:3] for x in [.10,.17,.23] for y in [-.05,.05] for z in [-.055,.015]])
            points=np.array(points)
            floor=float(points[:,2].min()+base_height-.025);minimum_floor=min(minimum_floor,floor)
            bodydist=np.linalg.norm(np.maximum(np.abs(points)-np.array([.3235,.125,.075]),0),axis=1)
            if bodydist.min()<.025: failures.append({'alpha':float(alpha),'reason':'trunk_clearance','distance':float(bodydist.min())})
            for leg in ['FR','FL','RR','RL']:
                for a,b in [(f'{leg}_hip',f'{leg}_thigh'),(f'{leg}_thigh',f'{leg}_calf'),(f'{leg}_calf',f'{leg}_foot')]:
                    if a not in legs or b not in legs: continue
                    aa=legs[a][:3,3];bb=legs[b][:3,3];ab=bb-aa
                    u=np.clip((points-aa)@ab/max(ab@ab,1e-12),0,1)
                    d=np.linalg.norm(points-(aa+u[:,None]*ab),axis=1)
                    if d.min()<.055: failures.append({'alpha':float(alpha),'reason':'leg_clearance','link':a,'distance':float(d.min())})
            if floor<.002: failures.append({'alpha':float(alpha),'reason':'floor_clearance','distance':floor})
        return {'safe':not failures,'minimum_floor':minimum_floor,'failures':failures[:6]}
