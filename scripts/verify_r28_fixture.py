"""Independent scalar radiance fixtures; numerical checks, not performer texture."""
import argparse
import numpy as np
from numpy.polynomial.legendre import leggauss
from dataclasses import replace
from audit_r24_source import write, ROOT, sha
from connected_eye import ConnectedEye, trace_connected
from corneal_transport import interface
from eye_appearance_fit import DIRECTIONS, light_basis, fit as old_fit
from eye_radiance_fit import sphere_gram, fit_radiance, angular_support, design_with_backing
from shot_paths import add_shot_argument, from_args


def controls():
    # Independent tensor quadrature for the exact sphere-mean product integral.
    z,w=leggauss(72); phi=(np.arange(144)+.5)*2*np.pi/144
    dirs=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),
        np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]),-1).reshape(-1,3)
    b=light_basis(dirs); weights=np.repeat(w/288,144)
    integral=b.T@(b*weights[:,None]); gram_error=float(np.max(np.abs(integral-sphere_gram())))
    assert gram_error<1e-12
    # Independent scalar truth, with patch identity and Fresnel varying separately
    # from the sampled direction. This intentionally provides layer excitation.
    n=360; z=1-2*(np.arange(n)+.5)/n; phi=np.arange(n)*2.399963229728653
    d=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],1)
    material=np.array([[.06,.1,.08],[.12,.07,.03],[.04,.08,.14]])
    lights=np.zeros((53,3)); lights[0]=[.12,.1,.08]; lights[5]=[1.2,.8,.4]; lights[39]=[.3,.7,1.1]
    rows=[]; truth=[]; layer_truth=[]
    for i,direction in enumerate(d):
        # Scalar dot/exponential loop avoids the production vectorized basis.
        radiance=lights[0].copy()
        for concentration,base in ((8,1),(32,27)):
            for j,axis in enumerate(DIRECTIONS):
                dot=sum(float(direction[k])*float(axis[k]) for k in range(3))
                radiance+=lights[base+j]*np.exp(concentration*(dot-1))
        for f in (.025,.12,.35):
            patch=i%3; a=np.zeros(3); a[patch]=1-f
            rows.append(np.r_[a,f*light_basis(direction)])
            layer_truth.append([(1-f)*material[patch],f*radiance])
            truth.append((1-f)*material[patch]+f*radiance)
    x=np.array(rows); y=np.array(truth); expected=np.array(layer_truth)
    train=np.repeat(np.arange(n)%5!=0,3); test=~train
    co=fit_radiance(x[train],y[train],1e-6,3)
    pred=x[test]@co; sum_error=float(np.max(np.abs(pred-y[test])))
    layer=np.stack([x[test,:3]@co[:3],x[test,3:]@co[3:]],1)
    layer_error=float(np.max(np.abs(layer-expected[test])))
    assert sum_error<.005 and layer_error<.015
    duplicate=fit_radiance(np.tile(x[train],(2,1)),np.tile(y[train],(2,1)),1e-6,3)
    exposure=fit_radiance(x[train],y[train]*7,1e-6,3)
    duplicate_error=float(np.max(np.abs(duplicate-co)))
    exposure_error=float(np.max(np.abs(exposure-7*co)))
    assert duplicate_error<1e-8 and exposure_error<1e-8
    # A tiny sampled tail can fit with enormous radiance in the old objective.
    tail=np.zeros((30,342)); tail[:,0]=.98; tail[:,290]=1e-9
    observed=np.tile([.1,.12,.08],(len(tail),1))
    query=tail[:1].copy(); query[:,290]=.02
    old=old_fit(tail,observed,1e-4); new=fit_radiance(tail,observed,1e-6,289)
    old_max=float((query@old).max()); new_max=float((query@new).max())
    assert old_max>1000 and new_max<1
    known=np.tile([0.,0.,1.],(10,9,1))
    unseen=np.tile([1.,0.,0.],(10,9,1))
    assert angular_support([known,known],known)[0].all()
    assert not angular_support([np.tile(known,(5,1,1))],known)[0].any()
    assert not angular_support([known,known],unseen)[0].any()
    mixed=known.copy(); mixed[:,0]=[1.,0.,0.]
    assert not angular_support([known,known],mixed)[0].any()
    # A withheld bright SG lobe with known scalar truth. A stable extrapolation
    # is still wrong; the unobserved direction must abstain rather than earn repair.
    lobe_train=np.zeros((20,54)); lobe_train[:,0]=.98
    lobe_train[:,1:]=.02*light_basis(np.tile([0.,0.,1.],(20,1)))
    lobe_query=np.r_[.98,.02*light_basis(np.array([1.,0.,0.]))][None]
    training_value=.98*.08+.02*(.1+20*np.exp(-32))
    query_truth=.98*.08+.02*(.1+20.)
    lobe_co=fit_radiance(lobe_train,np.full((20,3),training_value),1e-6,1)
    lobe_error=float(np.max(np.abs(lobe_query@lobe_co-query_truth)))
    assert lobe_error>.3 and not angular_support([known,known],unseen[:1])[0].any()
    # Deliberately unobservable layers: identical totals, different removal output.
    a1,l1,a2,l2,f=.08,1.,.1,.02,.02
    residual=abs((1-f)*a1+f*l1-((1-f)*a2+f*l2))
    assert residual<1e-14 and abs((1-f)*(a1-a2))>.019
    # Pupil aperture: scalar independent interface/plane hit and signed offset.
    eye=ConnectedEye(lid_height=13.,lid_width=13.)
    yy,xx=np.mgrid[30:66,30:66]; xy=np.stack([xx.ravel(),yy.ravel()],1)
    expected_changes=[]
    for offset in ([.6,0],[-.6,0],[0,.6],[0,-.6]):
        t=trace_connected(xy,eye,pupil_offset=offset)
        expected_pupil=[]
        for q,n,ray,cornea in zip(t['point'],t['normal'],t['ray'],t['cornea']):
            td,_,_=interface(ray@eye.rot,n,eye.ior)
            ti=(eye.iris_depth-q[2])/td[2]
            point=q+ti*td
            expected_pupil.append(bool(cornea and np.hypot(point[0]-offset[0],point[1]-offset[1])<eye.pupil_radius))
        assert np.array_equal(t['pupil'],expected_pupil)
        expected_changes.append(int(np.count_nonzero(t['pupil']!=trace_connected(xy,eye)['pupil'])))
    assert min(expected_changes)>0
    closed=replace(eye,lid_height=0.)
    _,valid,_,_=design_with_backing(xy,closed,[.6,0],(3000.,3000.),(47.5,47.5))
    assert not valid.any()
    # Direct BOX versus nine separately evaluated samples, including aperture crossings.
    points=np.array([[48,48],[51,48],[44,48],[44.5,48]],float)
    matrix,valid,_,fraction=design_with_backing(points,eye,[.6,0],(3000.,3000.),(47.5,47.5))
    from eye_appearance_fit import material_basis
    independent=[]
    for p in points:
        row=np.zeros(343)
        for dy in (-1/3,0,1/3):
            for dx in (-1/3,0,1/3):
                t=trace_connected(p+[dx,dy],eye,pupil_offset=[.6,0])
                material_row=material_basis(t['iris_xy'][None])[0][0]
                row+=np.r_[(1-t['fresnel'])*material_row*t['iris'],
                    (1-t['fresnel'])*t['pupil'],t['fresnel']*light_basis(t['reflected'])]/9
        independent.append(row)
    box_error=float(np.max(np.abs(matrix-independent)))
    mixed_pixels=int(((fraction>0)&(fraction<1)).sum())
    assert valid.all() and box_error<1e-14 and mixed_pixels>0
    return dict(sphere_gram_quadrature_max_error=gram_error,
        excited_fixture_sum_max_error=sum_error,excited_fixture_layer_max_error=layer_error,
        duplicate_observation_coefficient_max_error=duplicate_error,
        exposure_scaled_coefficient_max_error=exposure_error,
        unseen_tail_old_max_linear=old_max,unseen_tail_regularized_max_linear=new_max,
        angular_positive=10,unseen_direction_rejections=10,single_frame_rejections=10,
        one_bad_box_node_rejections=10,ambiguous_sum_error=residual,
        withheld_bright_lobe_rejected=True,withheld_bright_lobe_prediction_max_error=lobe_error,
        ambiguous_transmitted_difference=abs((1-f)*(a1-a2)),
        angular_coverage_does_not_resolve_ambiguity=True,
        signed_pupil_offset_changed_pixels=expected_changes,closed_rejected=int(len(xy)),
        pupil_box_max_error=box_error,mixed_pupil_box_pixels=mixed_pixels,
        clean_performer_material_validated=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    result=controls(); result['code_sha256']=sha(ROOT/'scripts/verify_r28_fixture.py')
    write(shot.logs/'r28_fixture.json',result); print('R28_FIXTURE',result,flush=True)


if __name__=='__main__': main()
