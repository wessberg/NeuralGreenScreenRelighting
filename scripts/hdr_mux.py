"""Insert standard HEVC HDR10 SEI metadata into NVENC Annex B output, then mux AAC.

SEI payload 137: mastering_display_colour_volume, 24 bytes, GBR primaries.
SEI payload 144: content_light_level_info, 4 bytes, measured CLL/FALL rounded upward.
NAL header: prefix SEI (39), layer 0, temporal_id_plus1=1. RBSP emulation prevention.
Metadata is inserted after parameter sets and before every random-access picture.
"""
from pathlib import Path
import struct,json,re,math,subprocess
ROOT=Path(__file__).resolve().parents[1]
FFMPEG=next((ROOT/'dependencies/ffmpeg').rglob('ffmpeg.exe'))

def escape_rbsp(data):
    result=bytearray();zeros=0
    for b in data:
        if zeros>=2 and b<=3:result.append(3);zeros=0
        result.append(b);zeros=zeros+1 if b==0 else 0
    return bytes(result)

def main():
    metrics=json.loads((ROOT/'logs/hdr_luminance.json').read_text())
    cll=math.ceil(metrics['max_cll_nits']);fall=math.ceil(metrics['max_fall_nits'])
    # Rec2020: G(.170,.797), B(.131,.046), R(.708,.292); D65(.3127,.3290).
    mastering=struct.pack('>8H2I',8500,39850,6550,2300,35400,14600,15635,16450,10000000,50)
    payload=bytes([137,24])+mastering+bytes([144,4])+struct.pack('>2H',cll,fall)+b'\x80'
    sei=b'\x00\x00\x00\x01\x4e\x01'+escape_rbsp(payload)
    data=(ROOT/'intermediate/encoded_pq.hevc').read_bytes()
    offsets=list(re.finditer(b'\x00\x00\x00\x01|\x00\x00\x01',data))
    patched=bytearray();previous=0;inserted=0
    for match in offsets:
        header=match.end()
        naltype=(data[header]>>1)&63
        if 16<=naltype<=23:
            patched.extend(data[previous:match.start()]);patched.extend(sei);previous=match.start();inserted+=1
    patched.extend(data[previous:])
    if not inserted:raise ValueError('No HEVC random-access frames found')
    elementary=ROOT/'intermediate/hdr10_metadata.hevc';elementary.write_bytes(patched)
    output=ROOT/'final/Night_Listening_Room_4K_HDR10.mp4'
    # Timing, derived from the shot rather than assumed. The mov timescale is the frame-rate
    # NUMERATOR so one tick is exactly one frame duration (den ticks): 30/1 -> 30, 30000/1001 -> 30000.
    # A 1000x-finer timescale cannot represent the B-frame (bf=3) composition offset exactly and the
    # muxer then pads the track by 400/30000 s, which trims 2 frames behind an edit list
    # (nb_read_frames 298) and reports avg_frame_rate 22500/751. composite.py encodes with -bf 0, so
    # there is no reordering here and no negative DTS to correct: -avoid_negative_ts make_zero would
    # only shift the track by one tick and desync the audio by a frame. -t applies to the AUDIO input
    # only, so it trims the long source read and never the video.
    shot=json.loads((ROOT/'shot.json').read_text())
    num,den=(int(x) for x in shot['source_frame_rate'].split('/'))
    frame_count=len(list((ROOT/'intermediate/linear_4k').glob('*.exr')))
    duration=f'{frame_count*den/num:.6f}'
    cmd=[str(FFMPEG),'-hide_banner','-y','-fflags','+genpts','-r',f'{num}/{den}','-i',str(elementary),
         '-t',duration,'-i',str(ROOT/'source/plate.mp4'),
         '-map','0:v:0','-map','1:a:0','-c:v','copy','-tag:v','hvc1','-c:a','aac','-b:a','192k',
         '-color_range','tv','-color_primaries','bt2020','-color_trc','smpte2084','-colorspace','bt2020nc',
         '-muxdelay','0','-muxpreload','0',
         '-movflags','+faststart','-video_track_timescale',str(num),
         '-metadata','title=Night Listening Room | Virtual Production Proof of Concept',str(output)]
    with open(ROOT/'logs/final_mux.log','w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=log)
    (ROOT/'logs/hdr_metadata.json').write_text(json.dumps({'mastering_primaries':'Rec2020','white_point':'D65','mastering_min_nits':.005,'mastering_max_nits':1000,'max_cll_nits':cll,'max_fall_nits':fall,'sei_insertions':inserted,'mux_command':cmd,'output':str(output)},indent=2))
    print('HDR10_MASTER',output,'bytes',output.stat().st_size,'MaxCLL',cll,'MaxFALL',fall)

if __name__=='__main__':main()
