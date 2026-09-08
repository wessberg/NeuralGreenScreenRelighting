"""One-off read-only query: Cycles pass names, SSS pass handling, panorama camera attribute path."""
import bpy, json
s = bpy.context.scene
vl = bpy.context.view_layer
passes = {}
for attr in dir(vl):
    if attr.startswith('use_pass_'):
        try: passes[attr] = bool(getattr(vl, attr))
        except Exception: pass
cy = getattr(vl, 'cycles', None)
cycles_passes = {}
if cy:
    for attr in dir(cy):
        if attr.startswith('use_pass_') or attr in ('denoising_store_passes',):
            try: cycles_passes[attr] = bool(getattr(cy, attr))
            except Exception: pass
cam = s.camera.data
panorama = {'has_panorama_type_on_data': hasattr(cam, 'panorama_type'),
            'has_cycles_subgroup': hasattr(cam, 'cycles'),
            'panorama_type_value': getattr(cam, 'panorama_type', None),
            'type_enum': [i.identifier for i in cam.bl_rna.properties['type'].enum_items]}
if hasattr(cam, 'panorama_type'):
    panorama['panorama_type_enum'] = [i.identifier for i in cam.bl_rna.properties['panorama_type'].enum_items]
# Is there a dedicated subsurface pass, or is SSS folded into diffuse?
sss = [a for a in dir(vl) if 'subsurface' in a.lower() or 'sss' in a.lower()]
shadow_catcher = {'object_attr': hasattr(bpy.types.Object, 'is_shadow_catcher'),
                  'viewlayer_pass': hasattr(vl.cycles, 'use_pass_shadow_catcher') if cy else None}
result = {'blender': bpy.app.version_string,
          'engine': s.render.engine,
          'view_transform': s.view_settings.view_transform,
          'viewlayer_passes_true': sorted(k for k, v in passes.items() if v),
          'viewlayer_passes_available': sorted(passes),
          'cycles_passes_true': sorted(k for k, v in cycles_passes.items() if v),
          'cycles_passes_available': sorted(cycles_passes),
          'subsurface_related_attrs': sss,
          'shadow_catcher': shadow_catcher,
          'panorama': panorama}
from pathlib import Path
Path(r'C:\Users\Frede\Developer\blender-virtual-production-experiment\logs\blender_facts.json').write_text(json.dumps(result, indent=2))
