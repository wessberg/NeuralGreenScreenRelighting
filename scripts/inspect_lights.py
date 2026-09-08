"""Read-only lighting audit: light emitters, world, exposure and view transform."""
import bpy,json
from pathlib import Path
s=bpy.context.scene
r={'engine':s.render.engine,'film_exposure':s.view_settings.exposure,
   'view_transform':s.view_settings.view_transform,'look':s.view_settings.look,
   'film_transparent':s.render.film_transparent,'resolution':[s.render.resolution_x,s.render.resolution_y],
   'samples':getattr(s.cycles,'samples',None),'lights':[],'emissive_materials':[],'world':None}
for o in bpy.data.objects:
    if o.type=='LIGHT':
        d=o.data
        r['lights'].append({'name':o.name,'type':d.type,'energy':d.energy,
            'color':list(d.color),'location':list(o.location),
            'size':getattr(d,'size',None),'size_y':getattr(d,'size_y',None),
            'shape':getattr(d,'shape',None),'hide_render':o.hide_render,
            'visible_camera':o.visible_camera})
w=s.world
if w and w.use_nodes:
    bg=[n for n in w.node_tree.nodes if n.type=='BACKGROUND']
    r['world']={'name':w.name,'backgrounds':[{'strength':n.inputs['Strength'].default_value,
        'color':list(n.inputs['Color'].default_value)[:3] if not n.inputs['Color'].is_linked else 'linked'} for n in bg]}
for m in bpy.data.materials:
    if not m.use_nodes: continue
    for n in m.node_tree.nodes:
        if n.type=='EMISSION':
            st=n.inputs['Strength']
            r['emissive_materials'].append({'material':m.name,'strength':st.default_value if not st.is_linked else 'linked'})
        elif n.type=='BSDF_PRINCIPLED' and 'Emission Strength' in n.inputs:
            v=n.inputs['Emission Strength']
            if not v.is_linked and v.default_value>0:
                r['emissive_materials'].append({'material':m.name,'principled_emission':v.default_value})
result=r
(Path(bpy.data.filepath).resolve().parents[1]/'logs/lights_inspection.json').write_text(json.dumps(r,indent=2))
