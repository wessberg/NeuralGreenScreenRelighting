"""Blender-only property probe for the pixel-filter mismatch caught in round 21."""
import argparse
import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from shot_paths import add_shot_argument, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    shot = from_args(args)
    scene = bpy.context.scene
    scene.render.filter_size = .01
    before = dict(render_filter_size=scene.render.filter_size,cycles_filter_width=scene.cycles.filter_width)
    assert abs(before['render_filter_size']-.01)<1e-8
    assert abs(before['cycles_filter_width']-1.5)<1e-8
    scene.cycles.filter_width = .01
    after = dict(render_filter_size=scene.render.filter_size,cycles_filter_width=scene.cycles.filter_width)
    assert abs(after['cycles_filter_width']-.01)<1e-8
    result = dict(blender=bpy.app.version_string,after_render_property_only=before,
        after_cycles_property=after,pixel_filter_type=scene.cycles.pixel_filter_type,
        scope='Factory-startup property probe; first run was archived and rerendered with frozen intended .01 width')
    (shot.logs/'r21_filter_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('R21_FILTER_PROBE_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
