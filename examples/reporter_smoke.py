"""Fresh reporter key/EyeClean/normals with explicitly reused same-source Blender passes."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ngsr import run_eyeclean, run_stage
from ngsr.engines import file_sha256

PASSES = ("ratio_render", "r44_light_bulb", "r44_light_fill", "r42_eye_pass")


def link_tree(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)


def prepare(source, output, count):
    output.mkdir(parents=True)
    meta = json.loads((source / "shot.json").read_text(encoding="utf-8"))
    meta["shot_id"] = output.name
    meta["selection"]["frame_count"] = count
    meta["environment"]["dir"] = str((ROOT / meta["environment"]["dir"]).resolve())
    (output / "shot.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    frames = output / "source/frames"
    frames.mkdir(parents=True)
    for i in range(count):
        os.link(source / "source/frames" / f"{i:06d}.png", frames / f"{i:06d}.png")
    (output / "relight").mkdir()
    for path in (source / "relight").iterdir():
        if path.is_file():
            shutil.copy2(path, output / "relight" / path.name)
    for name in ("proxy", "skin_mask", "depth"):
        link_tree(source / "relight" / name, output / "relight" / name)
    (output / "logs").mkdir()
    for name in ("source_light.json", "stress_proxy.json"):
        shutil.copy2(source / "logs" / name, output / "logs" / name)
    for name in PASSES:
        link_tree(source / "composite" / name, output / "composite" / name)
    return meta


def review(source, output, count, ffmpeg):
    import cv2
    import numpy as np
    import torch
    from ngsr.eyes import EyeCleanPatches
    review_dir = output / "review"
    for name in ("panels", "comparison", "eyes"):
        (review_dir / name).mkdir(parents=True, exist_ok=True)
    patches = EyeCleanPatches(output / "composite/eyeclean")
    rows = []
    def display(linear):
        t = (linear if isinstance(linear, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(linear), device="cuda")).clamp(min=0)
        t = torch.where(t <= .0031308, t*12.92, 1.055*t.pow(1/2.4)-.055)
        return (t.clamp(0, 1)*255).round().byte().cpu().numpy()
    def tile(image, label, size=(640, 360)):
        scale = min(size[0] / image.shape[1], size[1] / image.shape[0])
        resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                     max(1, round(image.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        image = np.zeros((size[1], size[0], 3), np.uint8)
        x, y = (size[0] - resized.shape[1]) // 2, (size[1] - resized.shape[0]) // 2
        image[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.rectangle(image, (0, 0), (size[0], 30), (15, 15, 15), -1)
        cv2.putText(image, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .58, (240,240,240), 1, cv2.LINE_AA)
        return image
    for i in range(count):
        name = f"{i:06d}"
        raw = cv2.imread(str(output / "source/frames" / (name+".png")), -1)
        raw = (raw/257).astype(np.uint8) if raw.dtype == np.uint16 else raw
        fg = cv2.imread(str(output / "key/fg" / (name+".exr")), -1)[..., ::-1].copy()
        alpha = cv2.imread(str(output / "key/alpha" / (name+".exr")), -1)
        clean, sides, edited = patches.apply(fg, i, output / "key/fg" / (name+".exr"))
        fg_t = torch.as_tensor(fg,device="cuda")
        clean_t = torch.as_tensor(clean,device="cuda")
        changed = torch.any(clean_t.view(torch.int32) != fg_t.view(torch.int32),dim=-1)
        # The adapter's support count must equal actual edited pixels on this fixture.
        assert int(changed.sum()) == edited
        old = cv2.imread(str(source / "composite/r44_m50_s05" / (name+".png")))
        new = cv2.imread(str(output / "composite/release" / (name+".png")))
        normal = np.load(output / "relight/normals" / (name+".npz"))["normal"].astype(np.float32)
        n_t = torch.as_tensor(normal,device="cuda")
        norm_error = (torch.linalg.vector_norm(n_t,dim=-1)-1).abs()
        normal_preview = ((n_t[...,[0,2,1]]*torch.tensor([.5,.5,-.5],device="cuda")+.5)*255).clamp(0,255).byte().cpu().numpy()[..., ::-1].copy()
        # Preview-only fixed crop of this reporter fixture; never used for inference/masking.
        crop = (slice(235,335), slice(885,1125))
        source_eye = display(fg)[..., ::-1][crop]
        clean_eye = display(clean)[..., ::-1][crop]
        result_eye = new[crop]
        eyes = np.concatenate([tile(source_eye,"Source", (720,300)),
                               tile(clean_eye,"EyeClean r57", (720,300)),
                               tile(result_eye,"New Blender reflections", (720,300))], axis=1)
        cv2.imwrite(str(review_dir / "eyes" / (name+".png")), eyes)
        a_t = torch.as_tensor(alpha,device="cuda")[...,None]
        yy,xx = torch.meshgrid(torch.arange(alpha.shape[0],device="cuda"),torch.arange(alpha.shape[1],device="cuda"),indexing="ij")
        checker = torch.where(((xx//32+yy//32)%2).bool(),.18,.04)[...,None]
        keyed = display(fg_t*a_t+checker*(1-a_t))[..., ::-1]
        panel = np.concatenate([
            np.concatenate([tile(raw,"Reporter source"),
                            tile(keyed,"CorridorKey 2.0 + core/despill"),
                            tile(normal_preview,"Fresh Sapiens2 normals (camera view)")], axis=1),
            np.concatenate([tile(old,"Accepted R44 baseline"),
                            tile(new,"New key + EyeClean + R44 skin"),
                            tile(eyes,"Source / clean / relit eyes")],axis=1)],axis=0)
        cv2.imwrite(str(review_dir / "panels" / (name+".png")), panel)
        comparison = np.concatenate([tile(raw,"Source",(960,540)),
                                     tile(old,"Accepted baseline",(960,540)),
                                     tile(new,"Updated pipeline",(960,540))],axis=1)
        cv2.imwrite(str(review_dir / "comparison" / (name+".png")), comparison)
        rows.append({"frame":i,"source_sha256":file_sha256(output / "source/frames" / (name+".png")),
                     "eye_pixels_changed":edited,"repaired_sides":sorted(sides),
                     "normal_unit_error_p99":float(torch.quantile(norm_error.flatten(),.99)),
                     "linear_finite":bool(torch.isfinite(fg_t).all()),
                     "new_vs_baseline_mean_abs_display":float((torch.as_tensor(new,device="cuda").float()-torch.as_tensor(old,device="cuda").float()).abs().mean())})
    selected = sorted(set(min(i,count-1) for i in (0,25,52,73,127)))
    tiles = [cv2.imread(str(review_dir / "panels" / f"{i:06d}.png")) for i in selected]
    contact = np.concatenate([cv2.resize(im,(1280,480)) for im in tiles],axis=0)
    cv2.imwrite(str(review_dir / "contact-sheet.jpg"), contact,[cv2.IMWRITE_JPEG_QUALITY,95])
    subprocess.run([str(ffmpeg),"-hide_banner","-y","-framerate","24","-i",
        str(review_dir / "comparison/%06d.png"),"-frames:v",str(count),"-c:v","libx264",
        "-crf","12","-pix_fmt","yuv420p",str(review_dir / "comparison.mp4")],check=True)
    metadata = json.loads((output / "shot.json").read_text(encoding="utf-8"))
    subprocess.run([str(ffmpeg), "-hide_banner", "-y", "-framerate",
        metadata["source_frame_rate"], "-i", str(review_dir / "eyes/%06d.png"),
        "-frames:v", str(count), "-c:v", "libx264", "-crf", "12", "-pix_fmt",
        "yuv420p", str(review_dir / "eyes.mp4")], check=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-shot",type=Path,default=ROOT/"music_videos/reporter_test/shots/reporter_a")
    ap.add_argument("--output-shot",type=Path,default=ROOT/"runs/reporter-release")
    ap.add_argument("--frames",type=int,default=150)
    ap.add_argument("--key-python",type=Path,default=ROOT/"dependencies/CorridorKey2/.venv/Scripts/python.exe")
    ap.add_argument("--eyeclean-python",type=Path)
    ap.add_argument("--ffmpeg",type=Path,required=True)
    ap.add_argument("--resume",action="store_true")
    args = ap.parse_args()
    source,output = args.source_shot.resolve(),args.output_shot.resolve()
    if not 0 < args.frames <= 150:
        ap.error("Reporter fixture contains 150 frames")
    if output.exists() and not args.resume:
        ap.error("Output exists; use a fresh directory or --resume")
    t0=time.time()
    if not output.exists():
        meta=prepare(source,output,args.frames)
    else:
        meta=json.loads((output/"shot.json").read_text())
        if meta["selection"]["frame_count"] != args.frames:
            ap.error("Resume frame count differs")
    if not (output/"logs/key_run.json").exists():
        subprocess.run([sys.executable,"-B",str(ROOT/"scripts/key_plate.py"),"--shot",str(output),
            "--python",str(args.key_python.resolve()),"--ffmpeg",str(args.ffmpeg.resolve()),
            "--frames",str(args.frames),"--low-vram"],cwd=ROOT,check=True)
    eyes=output/"composite/eyeclean"
    if not (eyes/"ngsr.json").exists():
        engine=[str(args.eyeclean_python.resolve()),"-B","-m","eyeclean"] if args.eyeclean_python else None
        run_eyeclean(output/"key/fg",eyes,fps=meta["source_frame_rate"],engine=engine,
            on_event=lambda ev: print(json.dumps(ev),flush=True))
    if not (output/"logs/neural_normals.json").exists():
        run_stage("normals",output,repository=ROOT,options=["--frames",str(args.frames)])
    sclera = output/"composite/sclera"
    if not (sclera/"removal.json").exists():
        run_stage("sclera",output,repository=ROOT,options=["--eye-patches",str(eyes),"--name","sclera"])
    options=["--sclera-pass",str(sclera),"--render-dir",str(output/"composite/ratio_render"),"--name","release",
        "--only-output",",".join(map(str,range(args.frames))),"--eye-patches",str(eyes),
        "--light-passes",f"bulb={output/'composite/r44_light_bulb'},fill={output/'composite/r44_light_fill'}",
        "--corneal-dir",str(output/"composite/r42_eye_pass"),"--corneal-blur","0.35",
        "--corneal-matte","lids","--meso-sigma","0","--meso-strength","0.5",
        "--specular-model","physical","--specular","0.5","--roughness-skin","0.55",
        "--f0-skin","0.028","--specular-texture","2","--specular-clamp","8","--linear-output"]
    run_stage("composite",output,repository=ROOT,options=options)
    rows=review(source,output,args.frames,args.ffmpeg.resolve())
    report={"schema":"ngsr.reporter-smoke/1","frames":args.frames,"source_shot":str(source),
        "output_shot":str(output),"recomputed":["CorridorKey 2.0 key","EyeClean r57","Sapiens2 normals",
                                                "R44 meso/skin specular","accepted sclera cleanup","ratio/eye composition"],
        "reused":["same-source proxy geometry","face/eye tracks","source-light estimate",*PASSES],
        "seconds_this_invocation":time.time()-t0,"artistic_acceptance":"pending",
        "hair_removal":False,"per_frame":rows,
        "eye_report":json.loads((eyes/"RUN_REPORT.json").read_text()),
        "dependency_sources":json.loads((ROOT/"dependencies.lock.json").read_text())}
    (output/"smoke-report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Review:",output/"review/contact-sheet.jpg",flush=True)


if __name__ == "__main__":
    main()
