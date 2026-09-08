"""Write aligned edge controls without changing the key or original composites."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
import argparse

p = argparse.ArgumentParser(description=__doc__)
add_shot_argument(p)
a = p.parse_args()
s = from_args(a)
out = s.composite / 'r4_diagnostics'
out.mkdir(exist_ok=True)
report = []
for n in (51, 79):
    old = cv2.imread(str(s.composite / 'ratio_render' / f'{n:06d}.exr'), -1)
    nw = cv2.imread(str(s.composite / 'r4_no_walls' / f'{n:06d}.exr'), -1)
    alpha = cv2.imread(str(s.alpha(n)), -1)
    if alpha.ndim == 3: alpha = alpha[..., 0]
    fg = cv2.imread(str(s.fg(n)), -1)
    from color_pipeline import linear_to_srgb
    panels = []
    for label, im in [('plate', linear_to_srgb(fg * alpha[..., None])),
                      ('original', cv2.imread(str(s.composite / 'ratio' / f'{n:06d}.png')) / 255),
                      ('irradiance', linear_to_srgb(old[..., :3])),
                      ('no walls', linear_to_srgb(nw[..., :3]))]:
        crop = np.clip(im[520:850, 540:900] * 255, 0, 255).astype('uint8')
        cv2.putText(crop, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        panels.append(crop)
    cv2.imwrite(str(out / f'{n:06d}_shoulder.png'), np.concatenate(panels, 1))
    rim = (alpha > .05) & (cv2.erode((alpha > .99).astype('uint8'), np.ones((9,9), 'uint8')) == 0)
    core = cv2.erode((alpha > .99).astype('uint8'), np.ones((21,21), 'uint8')) > 0
    report.append(dict(frame=n, rim_mean_abs_wall_difference=float(np.abs(old[...,:3]-nw[...,:3])[rim].mean()),
                       core_mean_abs_wall_difference=float(np.abs(old[...,:3]-nw[...,:3])[core].mean())))
(out / 'edge_controls.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
