"""Independent Cycles textured nonperformer fixture; no analytic RGB generator."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from render_r20_transport import annulus, emitter, mesh
from corneal_transport import rotation
from shot_paths import add_shot_argument, from_args


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def texture(name, rgb):
    height, width = rgb.shape[:2]
    image = bpy.data.images.new(name, width=width, height=height, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    rgba = np.concatenate([rgb, np.ones((height, width, 1))], -1).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    material = emitter(name, (1,1,1))
    node = material.node_tree.nodes.new('ShaderNodeTexImage')
    node.image, node.interpolation, node.extension = image, 'Linear', 'EXTEND'
    material.node_tree.links.new(node.outputs['Color'], material.node_tree.nodes.get('Emission').inputs['Color'])
    return material


def uv_map(obj, half):
    uv = obj.data.uv_layers.new(name='physical plane xy')
    for polygon in obj.data.polygons:
        for index in polygon.loop_indices:
            point = obj.data.vertices[obj.data.loops[index].vertex_index].co
            uv.data[index].uv = ((point.x/half+1)/2, (point.y/half+1)/2)


def textures():
    a = (np.arange(512)+.5)/512*2-1
    x,y = np.meshgrid(a,a)
    r,theta = np.hypot(x,y), np.arctan2(y,x)
    fiber = .5+.21*np.sin(41*theta+13*r+1.17)+.13*np.cos(79*theta-19*r+.29)
    spot = np.exp(-((x-.23)**2+(y+.19)**2)/.016)
    iris = np.stack([.035+.065*fiber+.055*spot,
                     .075+.11*fiber-.025*spot,
                     .095+.13*fiber-.035*spot], -1)
    pupil = np.stack([.006+.004*(x+1)/2, .009+.003*(y+1)/2,
                      .013+.004*(1+np.sin(7*x+3*y+1.3))/2], -1)
    source = np.full((512,512,3), .08)
    source[(x>-.6)&(x<-.05)&(y>-.5)&(y<.1)] = [5,5,5]
    source[(x>.05)&(x<.65)&(y>.2)&(y<.65)] = [1.5,1.1,.7]
    target = np.full((512,512,3), .04)
    target[(x>.1)&(x<.65)&(y>-.3)&(y<.3)] = [5,2.8,1.2]
    target[(x<-.1)&(x>-.65)&(y>-.1)&(y<.35)] = [.8,1.6,3.5]
    return iris, pupil, source, target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    shot = from_args(args)
    out = shot.composite/'r23_eyes/cycles'
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
    scene.cycles.device, scene.cycles.samples = 'GPU', 4096
    scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
    scene.cycles.seed, scene.cycles.max_bounces = 23, 8
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
    scene.render.filter_size = 1.0
    # render.filter_size does not update the separate Cycles setting in Blender 5.2.
    # BOX silently forces width 1 in Cycles' Blender sync, regardless of this RNA value.
    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width = 1.0
    assert abs(scene.cycles.filter_width-1.0) < 1e-8
    bpy.ops.object.camera_add(location=(0,0,0), rotation=(math.pi,0,0))
    camera = bpy.context.object
    camera.data.lens, camera.data.sensor_width = 3000*36/64, 36
    camera.data.clip_start, camera.data.clip_end = .1, 5000
    scene.camera = camera
    black = emitter('black aperture and clean environment', (0,0,0))
    iris_rgb, pupil_rgb, source_rgb, target_rgb = textures()
    iris_material, pupil_material, source_material, target_material = [texture(name, rgb)
        for name,rgb in zip(('iris fibers', 'nonblack pupil', 'source environment', 'target environment'),
                           (iris_rgb, pupil_rgb, source_rgb, target_rgb))]
    glass = bpy.data.materials.new('single anterior dielectric interface')
    glass.use_nodes = True
    glass.node_tree.nodes.clear()
    bsdf = glass.node_tree.nodes.new('ShaderNodeBsdfGlass')
    bsdf.distribution = 'BECKMANN'
    bsdf.inputs['Roughness'].default_value = 0
    bsdf.inputs['IOR'].default_value = 1.376
    output = glass.node_tree.nodes.new('ShaderNodeOutputMaterial')
    glass.node_tree.links.new(bsdf.outputs[0], output.inputs['Surface'])
    segments, rings = 512, 96
    rmax = math.sqrt(2*7.8*2.18-.75*2.18**2)
    vertices = [(0,0,0)]
    for r in np.linspace(rmax/rings, rmax, rings):
        z = (7.8-math.sqrt(7.8**2-.75*r*r))/.75
        vertices.extend((r*math.cos(a),r*math.sin(a),z)
            for a in np.linspace(0,2*math.pi,segments,endpoint=False))
    faces = [(0,1+(i+1)%segments,1+i) for i in range(segments)]
    for ring in range(rings-1):
        start = 1+ring*segments
        faces.extend((start+i,start+(i+1)%segments,start+(i+1)%segments+segments,start+i+segments)
                     for i in range(segments))
    cornea = mesh('independent tessellated cornea', vertices, faces, glass)
    for polygon in cornea.data.polygons:
        assert polygon.normal.z < 0
        polygon.use_smooth = True
    iris = annulus('textured iris', (1.8,1.8), (5.85,5.85), 3, iris_material)
    pupil = annulus('textured nonzero pupil', (0,0), (1.8,1.8), 3, pupil_material)
    uv_map(iris,5.85)
    uv_map(pupil,1.8)
    plane = mesh('textured environment', [(-250,-250,1100),(250,-250,1100),
        (250,250,1100),(-250,250,1100)], [(0,1,2,3)], source_material)
    plane.visible_camera = False
    uv_map(plane,250)
    records = []
    protocol_hash = sha(ROOT/'docs/r23-eye-protocol.md')
    for i in range(24):
        phase = 2*math.pi*i/24+.47
        yaw,pitch = 13*math.sin(phase),7*math.cos(phase)
        height = 4*(1-math.exp(-((i-17)/1.4)**2))
        rot = Matrix(rotation(math.radians(yaw),math.radians(pitch)).tolist())
        for obj in (cornea,iris,pupil):
            obj.location = (0,0,1300)
            obj.rotation_euler = rot.to_euler()
        lid = annulus('opaque aperture', (8,height), (40,40),1299,black)
        record = dict(frame=i,yaw=yaw,pitch=pitch,lid_height=height,files={})
        for name,material in (('source',source_material),('clean',black),('target',target_material)):
            plane.data.materials[0] = material
            path = out/f'{i:06d}_{name}.exr'
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            assert path.is_file()
            record['files'][name] = dict(path=str(path),sha256=sha(path))
        records.append(record)
        bpy.data.objects.remove(lid, do_unlink=True)
        print('R23_FRAME_COMPLETE',i,flush=True)
    report = dict(protocol_sha256=protocol_hash,
        code_sha256={name:sha(ROOT/'scripts'/name) for name in
            ('render_r23_eyes.py','render_r20_transport.py','corneal_transport.py')},
        blender=bpy.app.version_string, samples=4096, device=scene.cycles.device,
        devices=[d.name for d in prefs.devices if d.use],persistent_data=False,
        pixel_filter_type=scene.cycles.pixel_filter_type,pixel_filter_width=scene.cycles.filter_width,
        texture_resolution=512,render_resolution=64,records=records,
        scope='Nonperformer independent mesh/path tracing and texture basis; shared simplified anatomy')
    (shot.logs/'r23_cycles.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('R23_CYCLES_COMPLETE',flush=True)


if __name__ == '__main__':
    main()

