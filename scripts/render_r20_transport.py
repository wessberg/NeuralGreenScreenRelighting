"""Independent Cycles checks of the simplified cornea and dual-ray lid occlusion.

Run Blender -b --factory-startup --python this_file -- --shot ...
Constant emitters, not performer imagery. Single interface, no posterior cornea.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from corneal_transport import rotation
from shot_paths import add_shot_argument, from_args


def emitter(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1)
    emission.inputs['Strength'].default_value = 1
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    return material


def mesh(name, vertices, faces, material):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def annulus(name, inner, outer, z, material, segments=512):
    # Both ellipse radii are tuples. Face order is irrelevant for opaque emitters.
    vertices = [(rx*math.cos(a), ry*math.sin(a), z) for rx,ry in (inner,outer)
                for a in np.linspace(0, 2*math.pi, segments, endpoint=False)]
    faces = [(i,(i+1)%segments,(i+1)%segments+segments,i+segments) for i in range(segments)]
    return mesh(name, vertices, faces, material)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    shot = from_args(args)
    out = shot.composite/'r20_eyes/cycles'
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 4096
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 20
    scene.cycles.max_bounces = 8
    scene.render.use_persistent_data = False
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = 0
    scene.view_settings.view_transform = 'Standard'
    scene.render.filter_size = .01
    bpy.ops.object.camera_add(location=(0,0,0), rotation=(math.pi,0,0))
    camera = bpy.context.object
    camera.data.lens = 3000*36/64
    camera.data.sensor_width = 36
    camera.data.clip_start, camera.data.clip_end = .1, 5000
    scene.camera = camera
    black = emitter('absorbing lids and pupil', (0,0,0))
    interior = emitter('known interior radiance', (.1,.2,.3))
    env = emitter('known reflected radiance', (.7,.5,.2))
    glass = bpy.data.materials.new('single anterior dielectric interface')
    glass.use_nodes = True
    glass.node_tree.nodes.clear()
    bsdf = glass.node_tree.nodes.new('ShaderNodeBsdfGlass')
    bsdf.distribution = 'BECKMANN'
    bsdf.inputs['Roughness'].default_value = 0
    bsdf.inputs['IOR'].default_value = 1.376
    output = glass.node_tree.nodes.new('ShaderNodeOutputMaterial')
    glass.node_tree.links.new(bsdf.outputs[0], output.inputs['Surface'])
    # Radial tessellation of the conic, with front-facing (-z) normals.
    segments, rings = 512, 96
    rmax = math.sqrt(2*7.8*2.18-.75*2.18**2)
    verts = [(0,0,0)]
    for r in np.linspace(rmax/rings, rmax, rings):
        z = (7.8-math.sqrt(7.8**2-.75*r*r))/.75
        verts.extend((r*math.cos(a),r*math.sin(a),z)
                     for a in np.linspace(0,2*math.pi,segments,endpoint=False))
    faces = [(0, 1+(i+1)%segments, 1+i) for i in range(segments)]
    for ring in range(rings-1):
        start = 1+ring*segments
        for i in range(segments):
            j = (i+1)%segments
            faces.append((start+i,start+j,start+j+segments,start+i+segments))
    cornea = mesh('conic cornea', verts, faces, glass)
    for polygon in cornea.data.polygons:
        # Both the apex fan and ring quads must face the camera (-local z).
        # Reversing only the quads swaps the refractive medium despite plausible RGB.
        assert polygon.normal.z < 0, 'Corneal face winding reverses the optical interface'
        polygon.use_smooth = True
    iris = annulus('interior iris', (1.8,1.8), (5.85,5.85), 3, interior)
    pupil = annulus('black pupil', (0,0), (1.8,1.8), 3, black)
    plane = mesh('finite environment', [(-250,-250,1100),(250,-250,1100),
                 (250,250,1100),(-250,250,1100)], [(0,1,2,3)], env)
    plane.visible_camera = False
    cases = [('front',0,0,4),('yaw_plus',12,0,4),('yaw_minus',-12,0,4),
             ('pitch_plus',0,8,4),('pitch_minus',0,-8,4),('closed',0,0,0),
             ('narrow',0,0,1.5)]
    records = []
    for name,yaw,pitch,height in cases:
        rot = Matrix(rotation(math.radians(yaw),math.radians(pitch)).tolist())
        for obj in (cornea,iris,pupil):
            obj.location = (0,0,1300)
            obj.rotation_euler = rot.to_euler()
        lid = annulus('opaque aperture '+name, (8,height), (40,40),1299,black)
        scene.render.filepath = str(out/f'{name}.exr')
        bpy.ops.render.render(write_still=True)
        records.append(dict(name=name,yaw=yaw,pitch=pitch,lid_height=height,
                            path=str(out/f'{name}.exr')))
        bpy.data.objects.remove(lid, do_unlink=True)
    report = dict(blender=bpy.app.version_string, samples=scene.cycles.samples,
        device=scene.cycles.device, devices=[d.name for d in prefs.devices if d.use],
        persistent_data=False, cases=records, emitter_interior=[.1,.2,.3],
        emitter_environment=[.7,.5,.2], ior=1.376,
        scope='Independent mesh/path-traced validation of simplified optical transport; no performer')
    (shot.logs/'r20_cycles_transport.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('R20_CYCLES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
