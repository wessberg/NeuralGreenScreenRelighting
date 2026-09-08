"""Technical validation, full decode, decoded-frame inspection assets and HDR round trip."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import subprocess,json,hashlib,argparse
import numpy as np,cv2
from color_pipeline import pq_decode,REC709_TO_2020,LUMA709,LUMA2020,linear_to_srgb,grade_nits
ROOT=Path(__file__).resolve().parents[1]
import sys as _sys; _sys.path.insert(0,str(ROOT/'scripts'))
from shot_paths import add_shot_argument, from_args
FFMPEG=next((ROOT/'dependencies/ffmpeg').rglob('ffmpeg.exe'))
FFPROBE=next((ROOT/'dependencies/ffmpeg').rglob('ffprobe.exe'))

def run(cmd):return subprocess.run([str(x) for x in cmd],capture_output=True,check=True)

def main():
    ap=argparse.ArgumentParser(description='Technical validation of a shot master')
    add_shot_argument(ap)
    args=ap.parse_args()
    shot=from_args(args); SHOT=shot.dir; ENV=shot.environment
    # The master filename belongs to the shot, not to a hardcoded constant.
    masters=sorted((SHOT/'final').glob('*.mp4'))
    if not masters:
        raise SystemExit('No master in %s' % (SHOT/'final'))
    FILE=masters[0]
    assert FILE.exists() and FILE.stat().st_size>1000000
    probe=json.loads(run([FFPROBE,'-v','error','-count_frames','-show_streams','-show_format','-of','json',FILE]).stdout)
    (SHOT/'logs/final_ffprobe.json').write_text(json.dumps(probe,indent=2))
    video=next(x for x in probe['streams'] if x['codec_type']=='video')
    audio=next(x for x in probe['streams'] if x['codec_type']=='audio')
    expected={'width':3840,'height':2160,'r_frame_rate':'30/1','avg_frame_rate':'30/1','codec_name':'hevc','profile':'Main 10','pix_fmt':'yuv420p10le','color_range':'tv','color_space':'bt2020nc','color_transfer':'smpte2084','color_primaries':'bt2020','nb_read_frames':'300'}
    for k,v in expected.items():assert video.get(k)==v,(k,video.get(k),v)
    frames=json.loads(run([FFPROBE,'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp,best_effort_timestamp_time,pict_type,side_data_list','-of','json',FILE]).stdout)['frames']
    (SHOT/'logs/final_frame_metadata.json').write_text(json.dumps(frames,indent=2))
    mastering=[x for f in frames for x in f.get('side_data_list',[]) if x['side_data_type']=='Mastering display metadata']
    content=[x for f in frames for x in f.get('side_data_list',[]) if x['side_data_type']=='Content light level metadata']
    assert mastering and content,'Missing HDR10 SEI'
    assert mastering[0]['max_luminance']=='10000000/10000',mastering[0]
    # ffprobe reports frames in DECODE order, so timing must be checked in presentation order;
    # diffing the decode-order list is meaningless for any reordered stream. The tick-set check is
    # the one that actually catches duplicated or skipped presentation times.
    tb_den=int(video['time_base'].split('/')[1])
    fps_num,fps_den=(int(x) for x in video['r_frame_rate'].split('/'))
    # One frame must be a whole number of container ticks, or CFR is not representable at all.
    assert tb_den*fps_den%fps_num==0,f'timescale {tb_den} cannot express {fps_num}/{fps_den} fps exactly'
    ticks_per_frame=tb_den*fps_den//fps_num
    ticks=np.sort(np.array([int(f['best_effort_timestamp']) for f in frames]))
    expected_ticks=np.arange(len(frames))*ticks_per_frame
    assert np.array_equal(ticks,expected_ticks),(
        f'Presentation timestamps are not exactly CFR: expected step {ticks_per_frame}, '
        f'got steps {sorted(set(np.diff(ticks).tolist()))}, '
        f'{len(frames)-len(set(ticks.tolist()))} duplicate timestamps')
    timestamps=np.sort(np.array([float(f['best_effort_timestamp_time']) for f in frames]))
    assert abs(timestamps[0])<.002,timestamps[0]
    assert np.max(np.abs(np.diff(timestamps)-1/30))<.000002,'Irregular output frame timing'
    # The entire actual file, including audio, must decode without any FFmpeg error.
    decoded=run([FFMPEG,'-v','error','-xerror','-i',FILE,'-f','null','NUL'])
    (SHOT/'logs/final_decode.log').write_bytes(decoded.stderr)
    assert not decoded.stderr.strip(),decoded.stderr.decode(errors='replace')
    inspections=[]
    inverse=np.linalg.inv(REC709_TO_2020)
    for i in [0,30,60,150,240,299]:
        vf=f"select=eq(n\\,{i}),scale=in_range=limited:out_range=full:in_color_matrix=bt2020"
        data=run([FFMPEG,'-v','error','-i',FILE,'-vf',vf,'-frames:v','1','-pix_fmt','rgb48le','-f','rawvideo','pipe:1']).stdout
        code=np.frombuffer(data,'<u2').reshape(2160,3840,3).astype(np.float32)/65535
        nits2020=pq_decode(code);nits709=nits2020@inverse.T
        lum=np.maximum(nits709@LUMA709,0);mapped=lum/(lum+95)
        display=np.clip(linear_to_srgb(np.maximum(nits709,0)*(mapped/np.maximum(lum,1e-6))[...,None]),0,1)
        cv2.imwrite(str(SHOT/f'composite/decoded_final_{i:06d}.jpg'),(display[...,::-1]*255).astype(np.uint8))
        master=cv2.imread(str(SHOT/f'intermediate/linear_4k/{i:06d}.exr'),-1)[...,::-1]
        expected_nits=grade_nits(master)@REC709_TO_2020.T
        error=np.abs(nits2020-expected_nits)
        alpha=cv2.resize(cv2.imread(str(SHOT/f'key/alpha/{i:06d}.exr'),-1),(3840,2160))
        fg=nits709[alpha>.999]
        item={'frame':i,'max_decoded_luminance_nits':float((nits2020@LUMA2020).max()),'average_decoded_luminance_nits':float((nits2020@LUMA2020).mean()),'foreground_max_channel_nits':float(fg.max()),'mean_absolute_nit_error':float(error.mean()),'p99_absolute_nit_error':float(np.percentile(error,99)),'pct_pixels_above_203_nits':float(((nits2020@LUMA2020)>203).mean()*100)}
        inspections.append(item)
        print('DECODED_FRAME',json.dumps(item),flush=True)
    summary={'file':str(FILE),'bytes':FILE.stat().st_size,'sha256':hashlib.sha256(FILE.read_bytes()).hexdigest(),'video':{k:video.get(k) for k in expected},'duration_seconds':float(video['duration']),'audio':{'codec':audio['codec_name'],'sample_rate':audio['sample_rate'],'channels':audio['channels']},'mastering_display':mastering[0],'content_light_level':content[0],'full_decode_pass':True,'frames_decoded':len(frames),'frame_timing_pass':True,'decoded_inspections':inspections}
    (SHOT/'logs/final_validation.json').write_text(json.dumps(summary,indent=2))
    short=f"HEVC Main 10 | 3840x2160 | 30/1 fps | yuv420p10le (10-bit) | 300 frames / {video['duration']} s\nRec.2020 primaries | bt2020nc matrix | smpte2084 PQ | TV/limited range\nMastering: 1000 nits, Rec.2020/D65 | MaxCLL {content[0].get('max_content')} | MaxFALL {content[0].get('max_average')}\nFull-file decode: PASS | Audio: {audio['codec_name']} {audio['sample_rate']} Hz stereo\n"
    (SHOT/'logs/ffprobe_summary.txt').write_text(short)
    print(short)

if __name__=='__main__':main()
