"""Connected gross eye topology. Supplied geometry, NOT an anatomical estimator.

Millimetres; x right, y down, z away. A single effective refractive interface,
opaque spherical sclera, iris aperture and recessed diagnostic radiance backing.
The spherical lid shell is fixed in world axes while gaze rotates about the globe.
"""
from dataclasses import dataclass
import numpy as np
from corneal_transport import EyeOptics, intersect_conic, interface, rotation, unit


@dataclass(frozen=True)
class ConnectedEye:
    globe: tuple = (0., 0., 1313.)
    yaw: float = 0.
    pitch: float = 0.
    cornea_radius: float = 7.8
    eccentricity: float = .5
    limbus_radius: float = 5.85
    globe_radius: float = 12.
    iris_depth: float = 3.6
    pupil_radius: float = 1.8
    backing_depth: float = 5.
    ior: float = 1.376
    lid_radius: float = 13.8
    lid_width: float = 9.
    lid_height: float = 4.

    def __post_init__(self):
        numbers=[*self.globe,self.yaw,self.pitch,self.cornea_radius,self.eccentricity,
                 self.limbus_radius,self.globe_radius,self.iris_depth,self.pupil_radius,
                 self.backing_depth,self.ior,self.lid_radius,self.lid_width,self.lid_height]
        if not np.isfinite(numbers).all(): raise ValueError('Nonfinite eye geometry')
        if not (0 <= self.eccentricity < 1 and self.cornea_radius > self.limbus_radius >
                self.pupil_radius > 0 and self.globe_radius > self.limbus_radius and self.ior>1):
            raise ValueError('Invalid eye dimensions')
        if not (self.sag < self.iris_depth < self.backing_depth < self.centre_depth and
                self.lid_radius > self.centre_depth and 0 < self.lid_width < self.lid_radius and
                0 <= self.lid_height <= self.lid_width):
            raise ValueError('Crossing surfaces or invalid lid shell')

    @property
    def sag(self):
        p=1-self.eccentricity**2
        return (self.cornea_radius-np.sqrt(self.cornea_radius**2-p*self.limbus_radius**2))/p

    @property
    def centre_depth(self):
        return self.sag+np.sqrt(self.globe_radius**2-self.limbus_radius**2)

    @property
    def rot(self): return rotation(self.yaw,self.pitch)

    @property
    def apex(self): return np.asarray(self.globe)-np.array([0.,0.,self.centre_depth])@self.rot.T

    @property
    def optical(self):
        return EyeOptics(apex=tuple(self.apex),yaw=self.yaw,pitch=self.pitch,
            radius=self.cornea_radius,eccentricity=self.eccentricity,max_sag=self.sag,
            interior_depth=self.iris_depth,iris_radius=self.limbus_radius,
            pupil_radius=self.pupil_radius,ior=self.ior)


def sphere_hits(o,d,centre,radius):
    v=np.asarray(o)-centre
    b=np.sum(v*d,-1); c=np.sum(v*v,-1)-radius*radius
    disc=b*b-c; root=np.sqrt(np.maximum(disc,0))
    return -b-root,-b+root,disc>=0


def lid_blocked(o,d,eye):
    """Nearest positive hit on the anterior opaque spherical shell, excluding its hole."""
    t1,t2,hit=sphere_hits(o,d,np.asarray(eye.globe),eye.lid_radius)
    blocked=np.zeros(hit.shape,bool)
    for t in (t1,t2):
        q=np.asarray(o)+t[...,None]*d-np.asarray(eye.globe)
        aperture=np.zeros(hit.shape,bool) if eye.lid_height==0 else (
            (q[...,0]/eye.lid_width)**2+(q[...,1]/eye.lid_height)**2 < 1)
        blocked |= hit & (t>1e-6) & (q[...,2]<0) & ~aperture
    return blocked


def trace_connected(xy,eye,*,focal=(3000.,3000.),centre=(47.5,47.5),
                    environment_z=1100.,environment_half=250.):
    xy=np.asarray(xy,float)
    if xy.shape[-1]!=2 or not np.isfinite(xy).all() or min(focal)<=0:
        raise ValueError('Finite native pixel coordinates and positive focal required')
    ray=unit(np.concatenate([(xy-centre)/focal,np.ones((*xy.shape[:-1],1))],-1))
    d=ray@eye.rot; o=-eye.apex@eye.rot
    q,n,cap=intersect_conic(o,d,eye.optical)
    tc=np.sum((q-o)*d,-1)
    centre_local=np.array([0.,0.,eye.centre_depth])
    t1,t2,disc=sphere_hits(o,d,centre_local,eye.globe_radius)
    ts=np.full(cap.shape,np.inf)
    for t in (t1,t2):
        s=o+t[...,None]*d
        valid=disc&(t>0)&(s[...,2]>=eye.sag-1e-8)
        ts=np.minimum(ts,np.where(valid,t,np.inf))
    cap &= tc<ts
    blocked=lid_blocked(np.zeros_like(ray),ray,eye)
    cornea=cap&~blocked
    sclera=np.isfinite(ts)&~cap&~blocked
    td,rd,fresnel=interface(d,n,eye.ior)
    ti=(eye.iris_depth-q[...,2])/np.where(td[...,2]>1e-12,td[...,2],1e-12)
    interior=q+ti[...,None]*td
    ri=np.linalg.norm(interior[...,:2],axis=-1)
    # The iris annulus extends to its physical contact with the globe at its depth.
    iris_outer=np.sqrt(eye.globe_radius**2-(eye.iris_depth-eye.centre_depth)**2)
    g1,g2,gh=sphere_hits(q+td*1e-6,td,centre_local,eye.globe_radius)
    wall=np.full(cap.shape,np.inf)
    for g in (g1,g2):
        wall_point=q+td*1e-6+g[...,None]*td
        valid=gh&(g>0)&(wall_point[...,2]>=eye.sag-1e-8)
        wall=np.minimum(wall,np.where(valid,g,np.inf))
    iris=cornea&(ti>0)&(ti<wall)&(ri>=eye.pupil_radius)&(ri<=iris_outer)
    through=cornea&(ti>0)&(ti<wall)&(ri<eye.pupil_radius)
    tb=(eye.backing_depth-q[...,2])/np.where(td[...,2]>1e-12,td[...,2],1e-12)
    backing=q+tb[...,None]*td
    pupil=through&(tb<wall)&(tb>ti)
    internal_wall=cornea&~iris&~pupil
    world=q@eye.rot.T+eye.apex
    reflected=rd@eye.rot.T
    light_blocked=lid_blocked(world,reflected,eye)
    te=(environment_z-world[...,2])/np.where(np.abs(reflected[...,2])>1e-12,reflected[...,2],1e-12)
    env=world+te[...,None]*reflected
    env_uv=(env[...,:2]/environment_half+1)/2
    reflection=cornea&~light_blocked&(te>0)&(env_uv>=0).all(-1)&(env_uv<=1).all(-1)
    return dict(iris=iris,pupil=pupil,sclera=sclera,cornea=cornea,lid=blocked,
        internal_wall=internal_wall,reflection=reflection,light_blocked=light_blocked,
        point=q,normal=n,world=world,ray=ray,reflected=reflected,fresnel=fresnel,
        iris_xy=interior[...,:2],backing_xy=backing[...,:2],environment_uv=env_uv,
        pupil_aperture_xy=interior[...,:2],cap=cap,iris_outer=iris_outer)


def fixture_eye(i):
    phase=2*np.pi*i/24+.83
    return ConnectedEye(yaw=np.deg2rad(18*np.sin(phase)),pitch=np.deg2rad(10*np.cos(phase)),
        lid_height=4*(1-np.exp(-((i-16)/1.6)**2)))


def fixture_textures():
    a=(np.arange(512)+.5)/512*2-1; x,y=np.meshgrid(a,a)
    r,theta=np.hypot(x,y),np.arctan2(y,x)
    f=.5+.2*np.sin(31*theta+17*r+.91)+.1*np.cos(61*theta-11*r)
    iris=np.stack([.04+.07*f,.08+.1*f,.05+.065*f],-1)
    backing=np.stack([.011+.003*x,.018+.004*y,.025+.002*np.sin(9*x+5*y)],-1)
    old=np.full((512,512,3),.06); new=np.full_like(old,.03)
    old[(x>-.55)&(x<-.02)&(y>-.45)&(y<.15)]=[4,4.5,5]
    old[(x>.18)&(x<.7)&(y>.1)&(y<.5)]=[1.8,1.2,.8]
    new[(x>.12)&(x<.62)&(y>-.4)&(y<.25)]=[5,2.6,.9]
    new[(x>-.6)&(x<-.15)&(y>-.15)&(y<.5)]=[.7,1.8,4]
    return iris,backing,old,new
