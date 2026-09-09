"""Match the diagnostic Cycles fixture's finite cap and uncovered emissive iris.

This open-cap geometry is not anatomical sclera. Rays that miss its mesh can see
the iris plane directly. The inherited transport modeled only rays through glass.
"""
import numpy as np
from corneal_transport import trace, rotation


def trace_fixture(xy,eye,**kwargs):
    t=trace(xy,eye,**kwargs)
    rot=rotation(eye.yaw,eye.pitch)
    origin=-np.asarray(eye.apex)@rot
    direction=t['ray']@rot
    distance=(eye.interior_depth-origin[2])/direction[...,2]
    point=origin+distance[...,None]*direction
    radius=np.linalg.norm(point[...,:2],axis=-1)
    direct=(~t['hit'])&t['camera_open']&(distance>0)&(radius<=eye.iris_radius)
    t['interior_uv']=np.where(direct[...,None],(point[...,:2]/eye.iris_radius+1)/2,t['interior_uv'])
    t['interior_radius']=np.where(direct,radius,t['interior_radius'])
    t['fresnel']=np.where(direct,0.,t['fresnel'])
    t['visible']=t['visible']|direct
    t['pupil']=t['pupil']|(direct&(radius<eye.pupil_radius))
    t['iris']=t['iris']|(direct&(radius>=eye.pupil_radius))
    t['direct_interior']=direct
    return t
