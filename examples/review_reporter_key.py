"""Inspect keying separately from lighting, including raw CorridorKey 2 output."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., ::-1].copy()
    return torch.as_tensor(image, device='cuda')


def display(linear):
    linear = linear.clamp(min=0)
    rgb = torch.where(linear <= .0031308, linear * 12.92,
                      1.055 * linear.pow(1 / 2.4) - .055)
    return (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1].copy()


def labelled(image, label):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (245, 245, 245), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-shot', type=Path,
                        default=ROOT / 'music_videos/reporter_test/shots/reporter_a')
    parser.add_argument('--new-shot', type=Path, default=ROOT / 'runs/reporter-release')
    parser.add_argument('--frames', default='0,25,52,73,127')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Key comparison measurements require CUDA')
    out = args.new_shot / 'review/keying'
    out.mkdir(parents=True, exist_ok=True)
    # Fixed crops of this fixture for review only; never used to infer or alter a matte.
    crops = {'crown': (850, 65, 160, 160), 'hair_side': (815, 300, 160, 160),
             'hand': (1185, 280, 160, 160), 'coat': (570, 650, 160, 160)}
    labels = ('CorridorKey 1 + pipeline', 'CorridorKey 2 RAW', 'CorridorKey 2 + pipeline')
    audit = []
    for frame in map(int, args.frames.split(',')):
        name = f'{frame:06d}'
        raw = args.new_shot / 'key/corridorkey2'
        alphas = [read(args.old_shot / 'key/alpha' / (name + '.exr')),
                  read(raw / 'source_alpha_exr' / ('alpha_' + name + '.exr')),
                  read(args.new_shot / 'key/alpha' / (name + '.exr'))]
        fgs = [read(args.old_shot / 'key/fg' / (name + '.exr')),
               read(raw / 'source_fg_exr' / ('fg_' + name + '.exr')),
               read(args.new_shot / 'key/fg' / (name + '.exr'))]
        delta = (alphas[2] - alphas[0]).abs()
        suppressed = (alphas[1] - alphas[2]).clamp(min=0)
        audit.append({'frame': frame, 'image_pixels': delta.numel(),
                      'alpha_changed_over_0_01': int((delta > .01).sum()),
                      'raw_v2_alpha_suppressed_over_0_01': int((suppressed > .01).sum())})
        for background, level in [('white', 1.), ('grey', .18), ('black', 0.)]:
            images = [display(fg * a[..., None] + level * (1 - a[..., None]))
                      for fg, a in zip(fgs, alphas)]
            full = np.concatenate([labelled(im, label) for im, label in zip(images, labels)], axis=1)
            cv2.imwrite(str(out / f'{name}_{background}.jpg'), full,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
            for region, (x, y, w, h) in crops.items():
                tiles = [labelled(cv2.resize(im[y:y+h, x:x+w], (w*4, h*4),
                                             interpolation=cv2.INTER_NEAREST), label)
                         for im, label in zip(images, labels)]
                cv2.imwrite(str(out / f'{name}_{region}_{background}.png'),
                            np.concatenate(tiles, axis=1))
        matte = [(a * 255).round().byte().cpu().numpy() for a in alphas]
        cv2.imwrite(str(out / f'{name}_alpha.png'), np.concatenate(matte, axis=1))
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
