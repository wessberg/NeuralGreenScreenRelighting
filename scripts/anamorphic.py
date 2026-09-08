"""Anamorphic camera model for the artist's SIRUI Saturn 35 mm, 1.6x squeeze.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup \\
        --python scripts/anamorphic.py -- --validate

The artist shoots anamorphic and Blender renders spherically, so plate and CG do not share a
projection until something reconciles them. That reconciliation is here, in one place, because a
silent 1.6x error in horizontal scale is geometric rather than aesthetic: it would put the CG
background in a different perspective from the performer and no grade could fix it.

**The geometry works out exactly, which is worth stating because it looks like it should not.**

Delivery is 3840x1600 (2.40:1) *desqueezed*. Squeezed, that is 2.40 / 1.6 = 1.50:1 -- and a
full-frame gate is 36 x 24 mm, which is 1.50:1 exactly. So the full open gate is used, with no
crop, and:

    horizontal: spherical-equivalent focal = 35 / 1.6 = 21.875 mm
                hFOV = 2*atan(18 / 21.875)                    = 78.88 deg
    vertical:   true focal 35 mm
                vFOV = 2*atan(12 / 35)                        = 37.85 deg

Rendering the DESQUEEZED frame with a spherical camera at 21.875 mm on a 36 mm horizontal sensor
reproduces *both*, because the implied vertical sensor is 36 * 1600/3840 = 15 mm and
2*atan(7.5 / 21.875) = 37.85 deg -- the same angle, since 15 = 24/1.6 and 21.875 = 35/1.6.
No pixel-aspect trickery is needed for geometry.

**What does NOT come free is defocus**, in two separate ways:

* *Amount.* Blur diameter scales as focal^2 / N, so a camera at 21.875 mm renders far less blur
  than the 35 mm lens actually taking the picture. Matching it needs N scaled by
  (21.875/35)^2 = 1/1.6^2 = 0.3906 -- the reason an anamorphic gives the shallow depth of field
  of a long lens with the field of view of a wide one.
* *Shape.* Anamorphic bokeh is a vertical oval in the desqueezed image. Cycles has
  ``dof.aperture_ratio`` for exactly this. Which direction the ratio stretches is NOT asserted
  here: ``--validate`` renders defocused points and measures the ellipse, because a convention
  guess that happens to look plausible is precisely how this project has lost time before.

The DOF *magnitude* rule above is physics, but the T-stop the artist actually shoots at is not
recorded yet, and lens-design details (where the iris sits relative to the cylindrical elements)
shift real bokeh away from the ideal model. Treat ``aperture_fstop`` as a starting point to be
matched against a real defocused take, not as a measurement.
"""
import math, json, sys

SQUEEZE = 1.6
FOCAL_MM = 35.0
GATE_W_MM = 36.0
GATE_H_MM = 24.0
DELIVERY = (3840, 1600)


def model(focal_mm=FOCAL_MM, squeeze=SQUEEZE, gate_w=GATE_W_MM, gate_h=GATE_H_MM,
          delivery=DELIVERY, taking_fstop=2.9):
    """Every derived quantity, so callers never recompute a factor of 1.6 by hand."""
    eq = focal_mm / squeeze                                   # horizontal spherical equivalent
    hfov = 2 * math.atan(gate_w / (2 * eq))
    vfov = 2 * math.atan(gate_h / (2 * focal_mm))
    w, h = delivery
    implied_v_sensor = gate_w * h / w
    vfov_check = 2 * math.atan(implied_v_sensor / (2 * eq))
    return {
        'squeeze': squeeze,
        'taking_focal_mm': focal_mm,
        'render_focal_mm': eq,
        'render_sensor_width_mm': gate_w,
        'render_resolution': list(delivery),
        'squeezed_aspect': round((w / squeeze) / h, 4),
        'gate_aspect': round(gate_w / gate_h, 4),
        'gate_matches_squeezed_aspect': abs((w / squeeze) / h - gate_w / gate_h) < 1e-6,
        'hfov_deg': round(math.degrees(hfov), 3),
        'vfov_deg': round(math.degrees(vfov), 3),
        'vfov_deg_from_render_camera': round(math.degrees(vfov_check), 3),
        'vfov_consistent': abs(vfov - vfov_check) < 1e-9,
        'taking_fstop': taking_fstop,
        'render_fstop': round(taking_fstop / (squeeze ** 2), 4),
        'fstop_rule': 'blur diameter ~ focal^2 / N, so N scales by (render/taking focal)^2 = 1/squeeze^2',
        'bokeh': 'vertical oval, %.1f:1 in the desqueezed image' % squeeze,
    }


def apply_to_camera(cam_data, scn, m, aperture_ratio=None, focus_distance=None):
    """Configure a Blender camera for the desqueezed anamorphic render.

    ``aperture_ratio`` is left to the caller because its direction is a measured fact from
    ``--validate``, not something this function should assume.
    """
    cam_data.type = 'PERSP'
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = m['render_sensor_width_mm']
    cam_data.lens = m['render_focal_mm']
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = m['render_fstop']
    cam_data.dof.aperture_blades = 0
    if focus_distance is not None:
        cam_data.dof.focus_distance = focus_distance
    if aperture_ratio is not None:
        cam_data.dof.aperture_ratio = aperture_ratio
    scn.render.resolution_x, scn.render.resolution_y = m['render_resolution']
    scn.render.pixel_aspect_x = 1.0
    scn.render.pixel_aspect_y = 1.0
    return cam_data


def _validate():
    """Render defocused point sources and measure the bokeh ellipse for both ratio directions."""
    import bpy, bmesh, numpy as np
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / 'logs' / 'anamorphic'
    out_dir.mkdir(parents=True, exist_ok=True)
    m = model()

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = 256
    scn.cycles.use_denoising = False
    world = bpy.data.worlds.new('black')
    scn.world = world
    next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND'
         ).inputs['Strength'].default_value = 0.0

    # A small emissive sphere well behind the focus plane: its defocus disc IS the bokeh.
    me = bpy.data.meshes.new('pt')
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.004)
    bm.to_mesh(me); bm.free()
    pt = bpy.data.objects.new('pt', me)
    scn.collection.objects.link(pt)
    pt.location = (0.0, 6.0, 1.55)
    mat = bpy.data.materials.new('e')
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 4000.0
    nt.links.new(em.outputs['Emission'], nt.nodes['Material Output'].inputs['Surface'])
    pt.data.materials.append(mat)

    cam_data = bpy.data.cameras.new('anamorphic')
    cam = bpy.data.objects.new('anamorphic', cam_data)
    scn.collection.objects.link(cam)
    cam.location = (0.0, -1.3, 1.55)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scn.camera = cam
    scn.render.image_settings.media_type = 'IMAGE'
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'BW'
    scn.render.film_transparent = False

    results = {}
    for ratio in (1.0, SQUEEZE, 1.0 / SQUEEZE):
        apply_to_camera(cam_data, scn, m, aperture_ratio=ratio, focus_distance=1.3)
        # Small render: the disc is large and only its aspect is being measured.
        scn.render.resolution_x, scn.render.resolution_y = 960, 400
        path = out_dir / ('bokeh_ratio_%.4f.png' % ratio)
        scn.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(path))
        w, h = img.size
        a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)[..., 0]
        bpy.data.images.remove(img)
        thr = a.max() * 0.5
        ys, xs = np.where(a > thr)
        if not len(ys):
            results['%.4f' % ratio] = {'error': 'no disc above threshold'}
            continue
        # Pixel counts, converted to a physical aspect: the render has square pixels, so the
        # measured extents are directly comparable.
        wpx, hpx = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        results['%.4f' % ratio] = {
            'width_px': int(wpx), 'height_px': int(hpx),
            'bokeh_height_over_width': round(float(hpx) / float(wpx), 4),
            'render': str(path)}

    target = SQUEEZE
    best, best_err = None, 1e9
    for k, v in results.items():
        if 'bokeh_height_over_width' in v:
            err = abs(v['bokeh_height_over_width'] - target)
            if err < best_err:
                best, best_err = k, err
    report = {'model': m, 'measurements': results,
              'target_bokeh_height_over_width': target,
              'aperture_ratio_for_vertical_oval': best,
              'aperture_ratio_error': round(best_err, 4),
              'verdict': 'PASS' if best is not None and best_err < 0.25 else 'FAIL',
              'note': 'A vertical oval is the empirical signature of anamorphic bokeh in the '
                      'desqueezed image. The blur MAGNITUDE still has to be matched against a '
                      'real defocused take: the taking T-stop is not recorded yet.'}
    (ROOT / 'logs' / 'anamorphic_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if '--validate' in argv:
        _validate()
    else:
        print(json.dumps(model(), indent=2))
