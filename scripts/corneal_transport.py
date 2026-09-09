"""Explicit single-interface corneal transport; geometry is supplied, never inferred.

Camera/world axes: x right, y down, z away from camera; lengths in millimetres.
The conic x*x+y*y+(1-e*e)*z*z-2*R*z=0 is the anterior cornea approximation
also used by https://github.com/HadiZayer/eyenerf. This implementation is independent.
Homogeneous interior, planar iris and planar eyelid aperture are approximations.
"""
from dataclasses import dataclass

import numpy as np


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, sy*sp, sy*cp], [0, cp, -sp], [-sy, cy*sp, cy*cp]])


@dataclass(frozen=True)
class EyeOptics:
    apex: tuple = (0., 0., 1300.)
    yaw: float = 0.
    pitch: float = 0.
    radius: float = 7.8
    eccentricity: float = .5
    max_sag: float = 2.18
    interior_depth: float = 3.
    iris_radius: float = 5.85
    pupil_radius: float = 1.8
    ior: float = 1.376

    def __post_init__(self):
        if not np.isfinite([*self.apex, self.yaw, self.pitch, self.radius,
                self.eccentricity, self.max_sag, self.interior_depth,
                self.iris_radius, self.pupil_radius, self.ior]).all():
            raise ValueError('Nonfinite optical geometry')
        if not (self.radius > 0 and 0 <= self.eccentricity < 1 and self.ior > 1
                and 0 < self.pupil_radius < self.iris_radius
                and 0 < self.max_sag < self.interior_depth):
            raise ValueError('Invalid optical geometry')


def intersect_conic(origins, directions, eye):
    """Nearest forward intersection with the front cap, in cornea-local axes."""
    o, d = np.broadcast_arrays(np.asarray(origins, float), np.asarray(directions, float))
    p = 1-eye.eccentricity**2
    metric = np.array([1., 1., p])
    a = np.sum(d*d*metric, -1)
    b = 2*(np.sum(o*d*metric, -1)-eye.radius*d[..., 2])
    c = np.sum(o*o*metric, -1)-2*eye.radius*o[..., 2]
    disc = b*b-4*a*c
    t = (-b-np.sqrt(np.maximum(disc, 0)))/(2*a)
    xyz = o+t[..., None]*d
    valid = (disc >= 0) & (t > 0) & (xyz[..., 2] >= -1e-8) & (xyz[..., 2] <= eye.max_sag)
    normal = unit(xyz*metric-np.array([0., 0., eye.radius]))
    return xyz, normal, valid


def interface(d, n, ior):
    """Air -> homogeneous interior; exact unpolarized dielectric Fresnel."""
    cosine = np.clip(-np.sum(d*n, -1), 0, 1)
    eta = 1/ior
    transmitted_cosine = np.sqrt(1-eta**2*(1-cosine**2))
    refracted = eta*d+(eta*cosine-transmitted_cosine)[..., None]*n
    reflected = d+2*cosine[..., None]*n
    rs = (cosine-ior*transmitted_cosine)/(cosine+ior*transmitted_cosine)
    rp = (ior*cosine-transmitted_cosine)/(ior*cosine+transmitted_cosine)
    return unit(refracted), unit(reflected), (rs*rs+rp*rp)/2


def aperture_pass(origin, direction, plane_z, half_width, half_height):
    """A planar opaque lid with an elliptical hole, tested along the actual ray."""
    safe = np.where(np.abs(direction[..., 2]) > 1e-12, direction[..., 2], 1e-12)
    t = (plane_z-origin[..., 2])/safe
    point = origin+t[..., None]*direction
    if half_height <= 0 or half_width <= 0:
        return np.zeros(t.shape, bool)
    return (t > 0) & ((point[..., 0]/half_width)**2+(point[..., 1]/half_height)**2 < 1)


def trace(xy, eye, *, focal=(3000., 3000.), centre=(31.5, 31.5),
          lid_height=4., lid_width=8., lid_z=1299., environment_z=1100., environment_half=250.):
    """Return material/reflection coordinates and independent camera/light visibility.

    pixel centres have integer xy coordinates. fy need not equal fx: callers must
    explicitly supply their pixel aspect/camera convention. No capture-rig defaults.
    """
    xy = np.asarray(xy, np.float64)
    if xy.shape[-1] != 2 or not np.isfinite(xy).all() or min(focal) <= 0:
        raise ValueError('Finite native xy and positive focal lengths required')
    ray = unit(np.concatenate([(xy-centre)/focal, np.ones((*xy.shape[:-1], 1))], -1))
    rot, apex = rotation(eye.yaw, eye.pitch), np.asarray(eye.apex)
    local_d = ray@rot
    point, normal, hit = intersect_conic(-apex@rot, local_d, eye)
    transmitted, reflected_local, fresnel = interface(local_d, normal, eye.ior)
    ti = (eye.interior_depth-point[..., 2])/np.maximum(transmitted[..., 2], 1e-12)
    interior = point+ti[..., None]*transmitted
    radius = np.linalg.norm(interior[..., :2], axis=-1)
    point_world = point@rot.T+apex
    reflected = reflected_local@rot.T
    camera_open = aperture_pass(np.zeros_like(ray), ray, lid_z, lid_width, lid_height)
    incident_open = aperture_pass(point_world, reflected, lid_z, lid_width, lid_height)
    denom = np.where(np.abs(reflected[..., 2]) > 1e-12, reflected[..., 2], 1e-12)
    te = (environment_z-point_world[..., 2])/denom
    env = point_world+te[..., None]*reflected
    eye_visible = hit & camera_open & (ti > 0) & (radius <= eye.iris_radius)
    env_uv = (env[..., :2]/environment_half+1)/2
    reflection_visible = eye_visible & incident_open & (te > 0) & (env_uv >= 0).all(-1) & (env_uv <= 1).all(-1)
    return dict(interior_uv=(interior[..., :2]/eye.iris_radius+1)/2,
                environment_uv=env_uv, visible=eye_visible,
                pupil=eye_visible & (radius < eye.pupil_radius),
                iris=eye_visible & (radius >= eye.pupil_radius),
                reflection_visible=reflection_visible, camera_open=camera_open,
                incident_open=incident_open, fresnel=fresnel, ray=ray,
                local_point=point, local_normal=normal, local_direction=local_d,
                refracted=transmitted, reflected=reflected,
                point_world=point_world, hit=hit, interior_radius=radius)
