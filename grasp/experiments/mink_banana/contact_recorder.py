"""Read-only native PhysX pair impulses for independent grasp verification."""
import numpy as np
from pxr import PhysicsSchemaTools
from omni.physx import get_physx_simulation_interface

class ContactRecorder:
    def __init__(self):
        self.sums=np.zeros(2);self.events=[];self.errors=[]
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self.callback)
    def clear(self):
        self.sums[:]=0.;self.events=[]
    def callback(self,headers,data):
        try:
            for header in headers:
                a=str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                b=str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                other=b if '/Object' in a else a if '/Object' in b else None
                if other is None: continue
                jaw=0 if '/gripperStator' in other else 1 if '/gripperMover' in other else -1
                if jaw<0: continue
                impulse=0.;positions=[];normals=[];vectors=[]
                for i in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                    c=data[i];impulse+=float(np.linalg.norm(np.asarray(c.impulse)))
                    if len(positions)<3:
                        positions.append(list(c.position));normals.append(list(c.normal));vectors.append(list(c.impulse))
                self.sums[jaw]+=impulse
                if impulse>0: self.events.append({'jaw':jaw,'actor0':a,'actor1':b,'impulse_Ns':impulse,'positions':positions,'normals':normals,'impulse_vectors':vectors})
        except Exception as exc:
            if len(self.errors)<5: self.errors.append(type(exc).__name__+': '+str(exc))
    def forces(self,dt=.02): return (self.sums/dt).tolist()
    def close(self): self.subscription=None
