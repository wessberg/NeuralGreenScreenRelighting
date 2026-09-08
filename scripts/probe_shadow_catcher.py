"""Locate the shadow-catcher and visibility attribute paths on a real object in Blender 5.2.1."""
import bpy, json
from pathlib import Path
ob = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
res = {'object': ob.name if ob else None,
       'instance_has_is_shadow_catcher': hasattr(ob, 'is_shadow_catcher') if ob else None,
       'type_has_is_shadow_catcher': hasattr(bpy.types.Object, 'is_shadow_catcher'),
       'value': getattr(ob, 'is_shadow_catcher', None) if ob else None,
       'is_holdout': hasattr(ob, 'is_holdout'),
       'shadow_catcher_like_attrs': sorted(a for a in dir(ob) if 'catcher' in a.lower() or 'holdout' in a.lower()) if ob else [],
       'visibility_attrs': sorted(a for a in dir(ob) if a.startswith('visible_')) if ob else []}
Path(r'C:\Users\Frede\Developer\blender-virtual-production-experiment\logs\shadow_catcher_facts.json').write_text(json.dumps(res, indent=2))
