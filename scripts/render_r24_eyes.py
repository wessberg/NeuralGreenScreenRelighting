"""Independent Cycles tessellation of connected supplied eye geometry."""
import argparse
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from connected_eye import fixture_eye, fixture_textures
from render_r20_transport import emitter, mesh, annulus
from render_r23_eyes import texture, uv_map, sha
from shot_paths import add_shot_argument, from_args


def ring_surface(name,radii,zs,material,segments=512):
    vertices=[(r*math.cos(a),r*math.sin(a),z) for r,z in zip(radii,zs)
              for a in np.linspace(0,2*math.pi,segments,endpoint=False)]
    faces=[]
    for j in range(len(radii)-1):
        k=j*segments
        faces.extend((k+i,k+(i+1)%segments,k+segments+(i+1)%segments,k+segments+i)
                     for i in range(segments))
    obj=mesh(name,vertices,faces,material)
    for p in obj.data.polygons: p.use_smooth=True
    return obj


def lid_mesh(eye,material):
    angles=np.linspace(0,2*np.pi,512,endpoint=False)
    vertices=[]
    for blend in np.linspace(0,1,97):
        x=((1-blend)*eye.lid_width+blend*eye.lid_radius)*np.cos(angles)
        y=((1-blend)*eye.lid_height+blend*eye.lid_radius)*np.sin(angles)
        z=-np.sqrt(np.maximum(eye.lid_radius**2-x*x-y*y,0))
        vertices.extend(np.stack([x,y,z],-1)+eye.globe)
    faces=[(j*512+i,j*512+(i+1)%512,(j+1)*512+(i+1)%512,(j+1)*512+i)
           for j in range(96) for i in range(512)]
    return mesh('curved fixed lid shell',vertices,faces,material)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--frames',type=int,nargs='+',default=list(range(24)))
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    shot=from_args(args); out=shot.composite/'r24_eyes/cycles_oriented'; out.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; scene.render.engine='CYCLES'
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX'; prefs.get_devices()
    for device in prefs.devices: device.use=device.type=='OPTIX'
    scene.cycles.device='GPU'; scene.cycles.samples=4096
    scene.cycles.use_adaptive_sampling=scene.cycles.use_denoising=False
    scene.cycles.seed=24; scene.cycles.max_bounces=8
    scene.render.use_persistent_data=False
    scene.render.resolution_x=scene.render.resolution_y=96; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='OPEN_EXR'; scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='32'
    scene.world.use_nodes=True; scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=0
    scene.view_settings.view_transform='Standard'
    scene.cycles.pixel_filter_type='BOX'; scene.cycles.filter_width=1.
    bpy.ops.object.camera_add(location=(0,0,0),rotation=(math.pi,0,0))
    camera=bpy.context.object; camera.data.lens=3000*36/96; camera.data.sensor_width=36
    camera.data.clip_start=.1; camera.data.clip_end=5000; scene.camera=camera
    black=emitter('zero environment',(0,0,0)); sclera=emitter('sclera diagnostic radiance',(.32,.3,.27))
    lids=emitter('lid diagnostic radiance',(.12,.055,.035))
    mats=[texture(name,rgb) for name,rgb in zip(('iris','recessed nonblack backing','old scene','new scene'),fixture_textures())]
    glass=bpy.data.materials.new('effective anterior interface'); glass.use_nodes=True
    glass.node_tree.nodes.clear(); bsdf=glass.node_tree.nodes.new('ShaderNodeBsdfGlass')
    bsdf.distribution='BECKMANN'; bsdf.inputs['Roughness'].default_value=0
    bsdf.inputs['IOR'].default_value=1.376; output=glass.node_tree.nodes.new('ShaderNodeOutputMaterial')
    glass.node_tree.links.new(bsdf.outputs[0],output.inputs['Surface'])
    eye=fixture_eye(0)
    # Independent surface construction: cap radius determines sag, sphere meets its rim.
    radii=np.linspace(0,eye.limbus_radius,129); p=1-eye.eccentricity**2
    zs=(eye.cornea_radius-np.sqrt(eye.cornea_radius**2-p*radii*radii))/p
    cap=ring_surface('finite cornea',radii,zs,glass)
    assert all(p.normal.z<0 for p in cap.data.polygons if p.area>1e-12)
    angles=np.linspace(math.asin(eye.limbus_radius/eye.globe_radius),math.pi,257)
    globe=ring_surface('closed sclera',eye.globe_radius*np.sin(angles),
        eye.centre_depth-eye.globe_radius*np.cos(angles),sclera)
    outer=math.sqrt(eye.globe_radius**2-(eye.iris_depth-eye.centre_depth)**2)
    iris=annulus('physical iris aperture',(eye.pupil_radius,)*2,(outer,)*2,eye.iris_depth,mats[0])
    backing_outer=math.sqrt(eye.globe_radius**2-(eye.backing_depth-eye.centre_depth)**2)
    backing=annulus('recessed radiance backing',(0,0),(backing_outer,)*2,eye.backing_depth,mats[1])
    uv_map(iris,eye.limbus_radius); uv_map(backing,eye.limbus_radius)
    plane=mesh('finite scene emitter',[(-250,-250,1100),(250,-250,1100),(250,250,1100),(-250,250,1100)],[(0,1,2,3)],mats[2])
    plane.visible_camera=False; uv_map(plane,250)
    records=[]
    for i in args.frames:
        eye=fixture_eye(i); rot=Matrix(eye.rot.tolist())
        for obj in (cap,globe,iris,backing):
            obj.location=eye.apex; obj.rotation_euler=rot.to_euler()
        lid=lid_mesh(eye,lids); record=dict(frame=i,files={})
        for name,material in [('source',mats[2]),('clean',black),('target',mats[3])]:
            plane.data.materials[0]=material; path=out/f'{i:06d}_{name}.exr'
            scene.render.filepath=str(path); bpy.ops.render.render(write_still=True)
            record['files'][name]=dict(path=str(path),sha256=sha(path))
        records.append(record); bpy.data.objects.remove(lid,do_unlink=True)
        print('R24_FRAME_COMPLETE',i,flush=True)
    report=dict(protocol_sha256=sha(ROOT/'docs/r24-eye-protocol.md'),records=records,
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ('render_r24_eyes.py','connected_eye.py','render_r23_eyes.py','render_r20_transport.py')},
        blender=bpy.app.version_string,samples=4096,seed=24,device='OPTIX',
        filter='BOX',filter_width=1,persistent_data=False,denoising=False,
        scope='Supplied gross topology; single interface and radiance emitters, no real reconstruction')
    (shot.logs/'r24_cycles.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('R24_CYCLES_COMPLETE',flush=True)


if __name__=='__main__': main()
