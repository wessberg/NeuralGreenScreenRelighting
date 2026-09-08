"""Original procedural listening room; static camera; Cycles light transport and proxy shadows."""
import bpy, math, random, json, sys
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
TEST='--test' in sys.argv
random.seed(73)
# Datablock deletion is deterministic in both GUI and background MCP contexts.
for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
for m in bpy.data.materials:bpy.data.materials.remove(m)
s=bpy.context.scene
s.render.engine='CYCLES'
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='OPTIX';prefs.get_devices()
for d in prefs.devices:d.use=(d.type=='OPTIX');print('CYCLES_DEVICE',d.name,d.type,d.use)
s.cycles.device='GPU';s.cycles.samples=48 if TEST else 384
s.cycles.use_denoising=True;s.cycles.adaptive_threshold=.035 if TEST else .009
s.cycles.max_bounces=10;s.cycles.diffuse_bounces=5;s.cycles.glossy_bounces=5
s.cycles.seed=73;s.cycles.use_animated_seed=False
s.render.resolution_x=3840;s.render.resolution_y=2160;s.render.resolution_percentage=25 if TEST else 100
s.render.fps=30;s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_mode='RGBA';s.render.image_settings.color_depth='16';s.render.image_settings.exr_codec='ZIP'
s.view_settings.view_transform='AgX';s.view_settings.look='AgX - Medium High Contrast';s.view_settings.exposure=0
s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.12,.17,.26,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.12

def mat(name,color,rough=.5,metal=0,noise=0,scale=25):
    m=bpy.data.materials.new(name);m.use_nodes=True
    n=m.node_tree.nodes;l=m.node_tree.links;bs=n.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*color,1);bs.inputs['Roughness'].default_value=rough;bs.inputs['Metallic'].default_value=metal
    if noise:
        tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=scale;tex.inputs['Detail'].default_value=3
        coord=n.new('ShaderNodeTexCoord');l.new(coord.outputs['Generated'],tex.inputs['Vector'])
        bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.23;bump.inputs['Distance'].default_value=noise;l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
        ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(*(x*.7 for x in color),1);ramp.color_ramp.elements[1].color=(*(x*1.16 for x in color),1)
        l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],bs.inputs['Base Color'])
    return m

def wood(name,color):
    m=mat(name,color,.34);n=m.node_tree.nodes;l=m.node_tree.links
    tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=3.8;tex.inputs['Detail'].default_value=4;tex.inputs['Roughness'].default_value=.7
    coord=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY';mapping.inputs[1].default_value=(8,1,46);l.new(coord.outputs['Generated'],mapping.inputs[0]);l.new(mapping.outputs[0],tex.inputs[0])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(*(v*.45 for v in color),1);ramp.color_ramp.elements[1].color=(*(v*1.35 for v in color),1);l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],n['Principled BSDF'].inputs['Base Color'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.18;bump.inputs['Distance'].default_value=.0015;l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],n['Principled BSDF'].inputs['Normal'])
    return m

def emit(name,col,power):
    m=mat(name,col,.4);b=m.node_tree.nodes['Principled BSDF'];b.inputs['Emission Color'].default_value=(*col,1);b.inputs['Emission Strength'].default_value=power
    return m

def assign(o,m):
    o.data.materials.append(m);return o

def box(name,loc,dim,m,bevel=.015):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.view_layer.objects.active;o.name=name;o.dimensions=dim;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        b=o.modifiers.new('Soft manufactured edges','BEVEL');b.width=bevel;b.segments=3
        o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL')
    return assign(o,m)

def sphere(name,loc,scale,m):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=24,location=loc);o=bpy.context.view_layer.objects.active;o.name=name;o.scale=scale
    for p in o.data.polygons:p.use_smooth=True
    return assign(o,m)

def cylinder(name,loc,radius,depth,m,vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=radius,depth=depth,location=loc);o=bpy.context.view_layer.objects.active;o.name=name
    b=o.modifiers.new('Edge radius','BEVEL');b.width=.006;b.segments=3;o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    return assign(o,m)

def rod(name,a,b,r,m):
    o=cylinder(name,(Vector(a)+Vector(b))*.5,r,(Vector(b)-Vector(a)).length,m,24);o.rotation_euler=(Vector(b)-Vector(a)).to_track_quat('Z','Y').to_euler();return o

def area(name,loc,target,power,col,size,size_y=None):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.color=col;data.shape='RECTANGLE';data.size=size;data.size_y=size_y or size
    o=bpy.data.objects.new(name,data);s.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();return o

plaster=mat('Mineral plaster | warm graphite',(.17,.16,.14),.87,noise=.014,scale=75)
stone=mat('Honed limestone',(.25,.25,.22),.72,noise=.008,scale=40)
black=mat('Blackened steel',(.015,.019,.021),.3,.78)
brass=mat('Aged brass',(.32,.19,.065),.3,.78)
walnut=wood('Smoked walnut',(.15,.083,.038))
fabric=mat('Woven charcoal upholstery',(.075,.095,.096),.94,noise=.004,scale=170)
rugmat=mat('Flatweave rug | flax',(.19,.18,.15),.98,noise=.003,scale=210)
white=mat('Chalk glazed porcelain',(.52,.49,.42),.26,noise=.001,scale=13)

# Architecture. Deep-set rear window, oak floor with real joints, acoustic wall.
box('Floor underlay',(0,1,-.095),(7,12,.14),black)
for ix in range(27):
    for iy in range(6):
        x=-3.25+ix*.25;y=-4.2+iy*1.9+(ix%3)*.62
        pm=wood(f'Oak plank {ix}_{iy}',tuple(v*random.uniform(.83,1.12) for v in (.19,.12,.068)))
        box(f'Oak flooring {ix:02d}_{iy:02d}',(x,y,-.012),(.247,1.895,.045),pm,.003)
box('Rear right wall',(1.4,4.2,1.7),(3.4,.2,3.4),plaster)
box('Window bottom wall',(-1.85,4.2,.30),(3.1,.2,.60),plaster)
box('Window header',(-1.85,4.2,3.18),(3.1,.2,.44),plaster)
box('Left wall',(-3.4,.5,1.7),(.18,7.5,3.4),plaster)
box('Right wall',(3.25,.5,1.7),(.18,7.5,3.4),plaster)
box('Ceiling',(0,.5,3.45),(6.8,7.6,.16),plaster)
box('Rear skirting',(1.3,4.065,.08),(3.5,.04,.14),black,.004)
for x in [-3.28,-1.92,-.55]:box('Window mullion',(x,4.03,1.78),(.055,.12,2.38),black,.006)
for z in [.63,2.96]:box('Window transom',(-1.92,4.03,z),(2.78,.12,.045),black,.005)
box('Deep stone window sill',(-1.92,3.93,.62),(2.85,.43,.09),stone,.012)
# Outside city geometry, softened photographically by the lens.
night=emit('Night sky blue',(.055,.095,.19),.38)
box('Exterior blue atmosphere',(-4,42,8),(65,.05,35),night,0)
citymat=mat('Distant buildings',(.023,.034,.049),.9)
litwindows=[emit('City amber',(.95,.54,.23),3),emit('City blue white',(.48,.65,1),2.5)]
for k in range(30):
    x=-19+k*1.05;h=random.uniform(2.5,11);y=random.uniform(21,33)
    box('Distant tower',(x,y,h*.5),(.9,.8,h),citymat,.004)
    for zi in range(int(h/.42)):
        for xi in [-.28,0,.28]:
            if random.random()<.35:box('Distant illuminated window',(x+xi,y-.405,zi*.42+.20),(.075,.01,.14),random.choice(litwindows),0)
# Slightly reflective glazing, largely transparent.
glass=bpy.data.materials.new('Clear architectural glazing');glass.use_nodes=True;nt=glass.node_tree;nt.nodes.clear()
out=nt.nodes.new('ShaderNodeOutputMaterial');mix=nt.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.008
tr=nt.nodes.new('ShaderNodeBsdfTransparent');gl=nt.nodes.new('ShaderNodeBsdfGlossy');gl.inputs['Roughness'].default_value=.13
nt.links.new(tr.outputs[0],mix.inputs[1]);nt.links.new(gl.outputs[0],mix.inputs[2]);nt.links.new(mix.outputs[0],out.inputs['Surface'])
box('Window glazing',(-1.92,4.07,1.79),(2.67,.015,2.29),glass,0)
for k in range(29):box('Walnut acoustic batten',(1.04+k*.072,4.03,1.72),(.033,.075,3.2),walnut,.004)

# Low sofa on left, layers of actual upholstery with rolled edges.
for x in [-2.62,-.90]:
    for y in [2.02,2.77]:cylinder('Sofa brass foot',(x,y,.12),.027,.23,brass)
box('Sofa floating base',(-1.76,2.42,.29),(2.06,.97,.19),walnut,.05)
box('Sofa padded back',(-1.76,2.79,.79),(2.12,.24,.91),fabric,.11)
for x in [-2.68,-.84]:box('Sofa arm',(x,2.39,.57),(.22,.98,.62),fabric,.095)
for x in [-2.19,-1.33]:box('Sofa seat cushion',(x,2.31,.48),(.84,.76,.19),fabric,.08)
cushion=mat('Warm woven cushion',(.19,.125,.077),.91,noise=.004,scale=150)
o=box('Linen pillow',(-2.35,2.5,.79),(.44,.16,.43),cushion,.10);o.rotation_euler[1]=-.22;o.rotation_euler[0]=-.21

# Credenza and curated, restrained objects on right.
box('Credenza case',(1.9,3.5,.60),(1.72,.61,.78),walnut,.024)
for x in [1.15,2.65]:
    for y in [3.26,3.73]:cylinder('Credenza foot',(x,y,.16),.019,.29,black)
for x in [1.48,2.31]:box('Credenza door',(x,3.184,.60),(.79,.035,.72),walnut,.009)
for x in [1.83,1.96]:box('Recessed finger pull',(x,3.161,.7),(.012,.014,.12),black,.003)
bookcolors=[(.24,.18,.12),(.095,.115,.12),(.35,.31,.23),(.18,.065,.034)]
for j in range(5):
    bm=mat('Cloth book '+str(j),bookcolors[j%4],.84,noise=.002,scale=80)
    box('Art book '+str(j),(1.4,3.48,1.019+j*.032),(.44+random.uniform(-.03,.03),.30,.027),bm,.003)
cylinder('Porcelain vase foot',(2.29,3.53,1.01),.079,.035,white)
sphere('Porcelain vessel',(2.29,3.53,1.14),(.13,.13,.16),white)
cylinder('Porcelain vessel neck',(2.29,3.53,1.29),.060,.13,white)
for i in range(5):
    tip=(2.29+random.uniform(-.15,.16),3.53+random.uniform(-.10,.1),1.7+random.uniform(-.13,.17))
    rod('Dried botanical stem',(2.29,3.53,1.3),tip,.0025,walnut)
    for j in range(4):
        loc=Vector((2.29,3.53,1.3)).lerp(Vector(tip),.56+j*.12)
        o=sphere('Muted seed head',loc,(.045,.011,.02),cushion);o.rotation_euler[1]=random.uniform(-1,1)

# Floor lamp: a real diffuser, brass body, glowing shade, and local point emitter.
lampx,lampy=1.38,1.83
cylinder('Lamp weighted base',(lampx,lampy,.045),.24,.07,black)
cylinder('Brass lamp stem',(lampx,lampy,.81),.014,1.51,brass)
shade=mat('Warm linen lampshade',(.72,.49,.24),.85,noise=.002,scale=110)
bs=shade.node_tree.nodes['Principled BSDF'];bs.inputs['Emission Color'].default_value=(1,.52,.20,1);bs.inputs['Emission Strength'].default_value=.45
# Real transmission through linen lets the internal bulb create a soft luminance gradient.
nt=shade.node_tree;trans=nt.nodes.new('ShaderNodeBsdfTranslucent');trans.inputs['Color'].default_value=(.72,.57,.39,1)
mix=nt.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.68
nt.links.new(bs.outputs[0],mix.inputs[1]);nt.links.new(trans.outputs[0],mix.inputs[2]);nt.links.new(mix.outputs[0],nt.nodes['Material Output'].inputs['Surface'])
bpy.ops.mesh.primitive_cone_add(vertices=96,radius1=.30,radius2=.21,depth=.43,end_fill_type='NOTHING',location=(lampx,lampy,1.58));o=bpy.context.view_layer.objects.active;o.name='Open linen shade';assign(o,shade)
sol=o.modifiers.new('Shade thickness','SOLIDIFY');sol.thickness=.004
for z,r in [(1.365,.30),(1.795,.21)]:
    bpy.ops.mesh.primitive_torus_add(major_radius=r,minor_radius=.004,major_segments=80,minor_segments=8,location=(lampx,lampy,z));assign(bpy.context.view_layer.objects.active,brass)
light=bpy.data.lights.new('Warm practical bulb','POINT');light.energy=72;light.color=(1,.50,.23);light.shadow_soft_size=.14
o=bpy.data.objects.new('Warm practical bulb',light);s.collection.objects.link(o);o.location=(lampx,lampy,1.54)
sphere('Frosted bulb',(lampx,lampy,1.53),(.043,.043,.074),emit('Bulb emissive',(1,.60,.29),14))
# Seating area rug. Floor is visible behind lower edges, feet remain outside the source crop.
box('Woven rug',(0,.65,.023),(2.5,2.0,.013),rugmat,.008)
stripe=mat('Rug subtle border',(.11,.105,.085),1,noise=.002,scale=120)
for x in [-1.16,1.16]:box('Woven rug border',(x,.65,.031),(.025,1.88,.002),stripe,.001)

# Motivated cool key and off-camera broad fill. Export exact scene values to relighter.
area('Cool window key',(-2.75,-.65,2.52),(-.15,0,1.15),230,(.46,.67,1.0),2.4,2.1)
area('Window ambience',(-1.9,3.89,1.91),(-1.0,0,1.0),85,(.40,.59,1),2.5,2.1)
area('Warm practical soft bounce',(1.5,.7,1.75),(-.15,0,1.0),38,(1,.57,.30),1.15,1.0)
area('Camera soft fill',(.15,-3.15,2.0),(0,0,1.0),36,(.86,.90,1),2.5,2)
area('Ceiling bounced fill',(-.1,1.4,3.1),(0,1,0),30,(.69,.65,.57),3,2)
area('Credenza accent',(2.15,3.5,2.9),(2.1,3.5,1),12,(1,.60,.31),.8,.35)

# Static camera, photographed seated proportions. No animation or tracking.
bpy.ops.object.camera_add(location=(0,-4.3,1.02));cam=bpy.context.view_layer.objects.active;cam.name='Locked plate camera';cam.rotation_euler=(Vector((0,0,1.02))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=52;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';s.camera=cam
cam.data.dof.use_dof=True;cam.data.dof.focus_distance=4.3;cam.data.dof.aperture_fstop=5.6;cam.data.dof.aperture_blades=9

# Seated hidden proxy provides real soft occlusion in Cycles, never camera-visible.
proxy=mat('Hidden proxy neutral',(.14,.14,.14),.9)
proxy_objs=[]
for name,loc,scale in [
 ('head',(-.24,.055,1.41),(.13,.13,.19)),('torso',(-.23,.08,1.00),(.245,.15,.36)),
 ('left sleeve',(-.46,-.03,1.02),(.14,.16,.25)),('right sleeve',(.085,.025,.98),(.14,.15,.21)),
 ('lap',(-.13,-.02,.62),(.30,.27,.17)),('left leg',(-.26,.08,.32),(.16,.16,.32)),('right leg',(.18,.02,.28),(.12,.14,.30))]:
    o=sphere('PROXY '+name,loc,scale,proxy);o.visible_camera=False;o.visible_glossy=False;o.visible_transmission=False;proxy_objs.append(o)
o=box('PROXY seated support',(-.10,.23,.47),(.50,.45,.12),proxy,.025);o.visible_camera=False;o.visible_glossy=False;proxy_objs.append(o)
for x in [-.27,.08]:
    o=rod('PROXY stool leg',(x,.34,.43),(x,.39,.03),.023,proxy);o.visible_camera=False;o.visible_glossy=False;proxy_objs.append(o)

lighting={'camera':{'location':list(cam.location),'rotation':list(cam.rotation_euler),'matrix_world':[list(r) for r in cam.matrix_world],'lens_mm':52,'sensor_width_mm':36,'actor_distance_m':4.3,'height_m':1.02,'resolution':[3840,2160]},'lights':[], 'ambient_rgb':[.19,.205,.235], 'technique':'2.5D irradiance-ratio relighting with Video Depth Anything Small temporal geometry'}
bpy.context.view_layer.update()
lighting['camera']['matrix_world']=[list(r) for r in cam.matrix_world]
for o in bpy.data.objects:
    if o.type=='LIGHT':lighting['lights'].append({'name':o.name,'type':o.data.type,'position':list(o.location),'color':list(o.data.color),'power_w':o.data.energy,'direction':list(o.rotation_euler.to_matrix()@Vector((0,0,-1))),'size':getattr(o.data,'size',getattr(o.data,'shadow_soft_size',0))})
(ROOT/'blender/lighting.json').write_text(json.dumps(lighting,indent=2))
s['project_notes']='Original native geometry. Linear Rec709 EXR. Hidden proxy only for shadows; actor composited photographically. All shading controls exported to lighting.json.'
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender/listening_room.blend'))
prefix='test_' if TEST else ''
s.render.filepath=str(ROOT/f'blender/{prefix}background_shadow.exr');bpy.ops.render.render(write_still=True)
# Shadow ratio pass allows controlled compositing without displaying proxy geometry.
for o in proxy_objs:o.hide_render=True
s.render.filepath=str(ROOT/f'blender/{prefix}background_clean.exr');bpy.ops.render.render(write_still=True)
print('SCENE_RENDER_COMPLETE',flush=True)
