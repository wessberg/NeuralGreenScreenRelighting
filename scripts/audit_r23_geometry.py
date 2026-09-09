"""Fit separate real-eye optical proposals; predicted contours are NOT ground truth."""
import argparse
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from scipy.optimize import least_squares
from audit_r20_eyes import sha, write_json, write_image
from corneal_transport import EyeOptics, trace
from iris_observations import review_tile
from color_pipeline import linear_to_srgb, srgb_to_linear
from shot_paths import add_shot_argument, from_args

RADII=(7.2,7.8,8.4)


def optical(params,radius,origin,focal,centre):
    cx,cy,z,yaw,pitch,pupil=params
    xy=(np.array([cx,cy])+origin-centre)/focal*z
    return EyeOptics(apex=(*xy,z),yaw=yaw,pitch=pitch,radius=radius,pupil_radius=pupil)


def rays(points,params,radius,origin,focal,centre):
    eye=optical(params,radius,origin,focal,centre)
    return trace(points+origin,eye,focal=focal,centre=centre,
                 lid_z=eye.apex[2]-10,lid_width=1e5,lid_height=1e5)


def residual(params,radius,origin,focal,centre,iris,pupil):
    t=rays(np.concatenate([iris,pupil]),params,radius,origin,focal,centre)
    target=np.r_[np.full(len(iris),5.85),np.full(len(pupil),params[-1])]
    return (t['interior_radius']-target)*np.sqrt(np.prod(focal))/params[2]


def contours(params,radius,origin,focal,centre):
    theta=np.linspace(0,2*np.pi,96,endpoint=False)
    directions=np.stack([np.cos(theta),np.sin(theta)],-1)
    result=[]
    for target in (5.85,params[-1]):
        lo,hi=np.zeros(96),np.full(96,40.)
        for _ in range(28):
            mid=(lo+hi)/2
            points=params[:2]+mid[:,None]*directions
            values=rays(points,params,radius,origin,focal,centre)['interior_radius']
            lo=np.where(values<target,mid,lo); hi=np.where(values>=target,mid,hi)
        result.append(params[:2]+((lo+hi)/2)[:,None]*directions)
    return result


def fit_proposal(data,radius,origin,focal,centre):
    iris=data['limbus_boundary_xy'][::12]; pupil=data['pupil_boundary_xy'][::12]
    center=iris.mean(0); apparent=np.linalg.norm((iris-center)*[focal[1]/focal[0],1],axis=1).mean()
    z=np.sqrt(np.prod(focal))*5.85/apparent
    ratio=np.linalg.norm(pupil-pupil.mean(0),axis=1).mean()/np.linalg.norm(iris-center,axis=1).mean()
    bounds=([center[0]-8,center[1]-8,z*.5,-.8,-.8,.4],
            [center[0]+8,center[1]+8,z*2,.8,.8,4.8])
    results=[]
    for sign in (-1,0,1):
        initial=np.array([*center,z,sign*.2,-sign*.15,np.clip(ratio*5.85,.5,4.7)])
        result=least_squares(residual,initial,bounds=bounds,x_scale='jac',max_nfev=400,
            ftol=1e-10,xtol=1e-10,gtol=1e-10,
            args=(radius,origin,focal,centre,iris[::2],pupil[::2]))
        results.append(result)
    best=min(results,key=lambda r:np.dot(r.fun,r.fun)); params=best.x
    held=residual(params,radius,origin,focal,centre,iris[1::2],pupil[1::2])
    t=rays(np.concatenate([iris,pupil]),params,radius,origin,focal,centre)
    return params,dict(radius_mm=radius,parameters=params.tolist(),success=bool(best.success),
        evaluations=int(best.nfev),first_order_optimality=float(best.optimality),
        train_radial_rms_px=float(np.sqrt(np.mean(best.fun**2))),
        check_radial_rms_px=float(np.sqrt(np.mean(held**2))),
        actual_conic_hit_fraction=float(t['hit'].mean()),
        limbus_conic_hit_fraction=float(t['hit'][:len(iris)].mean()),
        pupil_conic_hit_fraction=float(t['hit'][len(iris):].mean()),
        complete_conic_support=bool(t['hit'].all()),
        fit_warning='Residual uses conic continuation outside finite cap; reject unless every contour ray hits',
        at_bound=bool(np.any(np.minimum(params-np.array(bounds[0]),np.array(bounds[1])-params)<1e-4)),
        starts=[dict(cost=float(r.cost),success=bool(r.success)) for r in results])


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    args=ap.parse_args(); shot=from_args(args)
    root=shot.relight/'neural_iris_observations'
    path=root/'manifest.json'; manifest=json.loads(path.read_text())
    config=json.loads((shot.dir/'shot.json').read_text())
    width,height=config['source_resolution']; camera=config['camera']
    fx=width*camera['lens_mm']/camera['sensor_width_mm']
    aspect=manifest['pixel_aspect']; focal=np.array([fx,fx*aspect]); centre=np.array([(width-1)/2,(height-1)/2])
    frames=np.linspace(0,len(manifest['frames'])-1,6).round().astype(int).tolist()
    out=shot.composite/'r23_geometry'; rows=[]
    for frame in frames:
        tiles=[]
        for side in ('left','right'):
            eye=manifest['frames'][frame]['eyes'][side]
            file=root/f'{frame:06d}_{side}.npz'; data=dict(np.load(file)); origin=data['origin_xy']
            h,w=data['linear_rgb'].shape[:2]
            complete=data['complete_ellipse_labels']>0
            visible=np.isin(data['visible_part_proposal'],[2,3])
            union=complete|visible
            row=dict(frame=frame,source_frame=manifest['frames'][frame]['source_frame'],side=side,
                archive_sha256=sha(file),origin_xy=origin.tolist(),profiles=[],
                complete_visible_proposal_iou=float((complete&visible).sum()/max(1,union.sum())),
                complete_pixels=int(complete.sum()),visible_proposal_pixels=int(visible.sum()),
                geometry_validated=False,lid_3d_validated=False,production_replacement=False)
            rgb=data['linear_rgb']; overlay=np.clip(linear_to_srgb(rgb)[...,::-1]*255,0,255).astype(np.uint8)
            overlay[visible]=cv2.addWeighted(overlay,.65,np.full_like(overlay,(30,80,15)),.35,0)[visible]
            query=data['pupil_boundary_xy'][::36]; reflected=[]
            for radius,color in zip(RADII,((0,100,255),(0,255,255),(255,160,0))):
                params,info=fit_proposal(data,radius,origin,focal,centre); row['profiles'].append(info)
                t=rays(query,params,radius,origin,focal,centre); reflected.append(t['reflected'])
                for curve in contours(params,radius,origin,focal,centre):
                    cv2.polylines(overlay,[np.rint(curve*16).astype(np.int32)],True,color,1,cv2.LINE_AA,4)
            angle=np.rad2deg(np.arccos(np.clip(np.sum(reflected[0]*reflected[-1],-1),-1,1)))
            row['profile_reflection_angle_deg']=dict(median=float(np.median(angle)),maximum=float(angle.max()),
                scope='Same proposed pupil-boundary pixels, radius endpoints; not measured ray accuracy')
            rows.append(row)
            label=f'{side} source {row["source_frame"]}'
            tiles.append(np.concatenate([review_tile(rgb,label+' EXACT SOURCE'),
                review_tile(srgb_to_linear(overlay[...,::-1]/255.),'3 radii / predicted geometry ONLY')],1))
            print('R23_GEOMETRY',frame,side,[p['check_radial_rms_px'] for p in row['profiles']],flush=True)
        board=np.concatenate(tiles,0)
        for repeat in range(8): write_image(out/'review'/f'{len(rows)//2*8-8+repeat:06d}.png',board)
    report=dict(camera=camera,focal_pixels=focal.tolist(),centre_pixels=centre.tolist(),pixel_aspect=aspect,
        manifest_sha256=sha(path),shot_config_sha256=sha(shot.dir/'shot.json'),frames=frames,records=rows,
        contour_split='Alternating samples from same predicted ellipse; NOT independent source or withheld RGB',
        priors='Fixed 5.85mm iris, conic eccentricity .5, depth 3mm, IOR 1.376; profiled radii are assumptions',
        lid_status='RITnet 2D visible-part comparison only; no 3D lid fit or scene-light visibility validation',
        optical_status='Fixed finite cap max_sag 2.18mm. Out-of-cap contour fits are rejected, not physical solutions.',
        full_eyenerf_run=False,geometry_validated=False,production_repairs=0,
        code_sha256=sha(Path(__file__)))
    write_json(shot.logs/'r23_geometry.json',report)
    print('R23_GEOMETRY_COMPLETE',flush=True)


if __name__=='__main__': main()
