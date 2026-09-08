import bpy,json
from pathlib import Path
r={'objects':len(bpy.data.objects),'camera':list(bpy.context.scene.camera.location),'selected':len([o for o in bpy.data.objects if o.select_get()]),'unassigned':[],'proxies':[],'sample':[]}
for o in bpy.data.objects:
    if o.type=='MESH' and len(o.data.materials)==0:r['unassigned'].append({'name':o.name,'position':list(o.location),'dimensions':list(o.dimensions),'visible_camera':o.visible_camera})
    if o.name.startswith('PROXY'):r['proxies'].append({'name':o.name,'visible_camera':o.visible_camera,'hide_render':o.hide_render})
    if o.name in ['Floor underlay','Rear right wall','Window glazing','Oak flooring 01_01','Exterior blue atmosphere']:r['sample'].append({'name':o.name,'location':list(o.location),'dimensions':list(o.dimensions),'materials':[m.name if m else None for m in o.data.materials]})
(Path(__file__).resolve().parents[1]/'logs/scene_inspection.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
