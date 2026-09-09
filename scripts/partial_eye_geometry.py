"""Conditional geometry families from partial observed arcs, with no RGB input.

Circular limbus and camera calibration are hypotheses. A compatible sampled
family is neither a posterior nor a certified anatomical/visibility estimate.
"""
import numpy as np
from scipy.optimize import least_squares
from corneal_transport import rotation
from connected_eye import ConnectedEye, trace_connected


def split_arc(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError('At least six finite ordered visible arc points required')
    third = len(points)//3
    check = np.arange(third, len(points)-third)
    train = np.r_[np.arange(third), np.arange(len(points)-third, len(points))]
    return points[train], points[check], train, check


def ring_world(parameters, yaw, pitch, focal, centre, radius=5.85, samples=360):
    cx, cy, z = parameters
    c = np.r_[(np.array([cx,cy])-centre)/focal*z, z]
    a = np.arange(samples)*2*np.pi/samples
    offsets = np.stack([radius*np.cos(a),radius*np.sin(a),np.zeros(samples)], -1)
    return c + offsets@rotation(yaw,pitch).T


def project(points, focal, centre):
    return np.asarray(points)[..., :2]/np.asarray(points)[..., 2:]*focal+centre


def curve(parameters, yaw, pitch, focal, centre, radius=5.85, samples=360):
    return project(ring_world(parameters,yaw,pitch,focal,centre,radius,samples),focal,centre)


def distances(points, contour):
    points, a = np.asarray(points,float), np.asarray(contour,float)
    delta = np.roll(a,-1,axis=0)-a
    u = np.clip(np.sum((points[:,None]-a)*delta,-1)/np.maximum(np.sum(delta*delta,-1),1e-20),0,1)
    return np.linalg.norm(points[:,None]-a-u[...,None]*delta,axis=-1).min(1)


def fit_family(train, focal, centre, radius=5.85, angles=range(-40,41,10)):
    """Train-only bounds, initialization and ranking. Native pixel distances."""
    train, focal, centre = map(lambda a: np.asarray(a,float), (train,focal,centre))
    if train.shape[0] < 4 or not np.isfinite(train).all() or np.min(focal)<=0:
        raise ValueError('Finite source arc and positive camera focal required')
    middle = (train.min(0)+train.max(0))/2
    span = max(np.ptp(train[:,0]),np.ptp(train[:,1]),4.)
    depth = np.sqrt(np.prod(focal))*radius/(span/2)
    lower = [*(middle-span),depth*.35]
    upper = [*(middle+span),depth*3]
    rows=[]
    for yd in angles:
        for pd in angles:
            yaw,pitch=np.deg2rad([yd,pd])
            def residual(params): return distances(train,curve(params,yaw,pitch,focal,centre,radius))
            result=least_squares(residual,[*middle,depth],bounds=(lower,upper),
                x_scale='jac',max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            d=residual(result.x)
            rows.append(dict(parameters=result.x.tolist(),yaw_deg=yd,pitch_deg=pd,
                train_distances_px=d.tolist(),train_rms_px=float(np.sqrt(np.mean(d*d))),
                train_max_px=float(d.max()),success=bool(result.success),nfev=int(result.nfev),
                at_bound=bool(np.min(np.minimum(result.x-lower,np.array(upper)-result.x))<1e-4),
                compatible=bool(result.success and d.max()<=2)))
    return rows


def connected_candidate(row, focal, centre, cornea_radius=7.8, pupil_radius=1.8):
    """Place the actual connected seam at the fitted perspective ring centre."""
    cx,cy,z=row['parameters']
    yaw,pitch=np.deg2rad([row['yaw_deg'],row['pitch_deg']])
    prototype=ConnectedEye(cornea_radius=cornea_radius,pupil_radius=pupil_radius,
                           yaw=yaw,pitch=pitch,lid_width=13.,lid_height=13.)
    ring_centre=np.r_[(np.array([cx,cy])-centre)/focal*z,z]
    globe=ring_centre+np.array([0.,0.,prototype.centre_depth-prototype.sag])@prototype.rot.T
    return ConnectedEye(globe=tuple(globe),yaw=yaw,pitch=pitch,
        cornea_radius=cornea_radius,pupil_radius=pupil_radius,lid_width=13.,lid_height=13.)


def pupil_distance(points, eye, focal, centre):
    """First-order native-pixel normal distance to the refracted aperture boundary.

    Only finite corneal hits count. This is a local residual, not an exact
    closest-contour distance, and never a pupil-radiance constraint.
    """
    points=np.asarray(points,float)
    if not len(points): return np.array([]),np.array([],bool)
    def radius_at(p):
        t=trace_connected(p,eye,focal=focal,centre=centre)
        return np.linalg.norm(t['pupil_aperture_xy'],axis=-1),t['cap']
    r,hit=radius_at(points)
    grad=np.stack([(radius_at(points+delta)[0]-radius_at(points-delta)[0])/.1
                   for delta in (np.array([.05,0]),np.array([0,.05]))],-1)
    return np.abs(r-eye.pupil_radius)/np.maximum(np.linalg.norm(grad,axis=-1),1e-8),hit


def conservative_open(shape, lid, reflection, uncertainty=2.):
    import cv2
    from audit_r24_source import polygon
    opening=polygon(shape,lid)
    inside=cv2.distanceTransform(opening.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    excluded=polygon(shape,reflection) if len(reflection) else np.zeros(shape,bool)
    outside=cv2.distanceTransform((~excluded).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Full BOX corner distance sqrt(.5), plus raster-center uncertainty sqrt(.5).
    margin=uncertainty+np.sqrt(2.)
    return opening,(inside>margin)&(outside>margin)


def route_family(xy, eyes, focal, centre, quadrature=3):
    """Nonempty candidate agreement only; no clean-material confidence is earned."""
    xy=np.asarray(xy,float).reshape(-1,2)
    if not eyes: return np.zeros(len(xy),bool),np.zeros(len(xy),bool)
    offsets=(np.arange(quadrature)+.5)/quadrature-.5
    samples=(xy[:,None,None,:]+np.stack(np.meshgrid(offsets,offsets),-1)).reshape(-1,2)
    all_iris=np.ones(len(xy),bool); all_pupil=all_iris.copy()
    for eye in eyes:
        t=trace_connected(samples,eye,focal=focal,centre=centre)
        all_iris &= t['iris'].reshape(len(xy),-1).all(1)
        all_pupil &= t['pupil'].reshape(len(xy),-1).all(1)
    return all_iris,all_pupil
