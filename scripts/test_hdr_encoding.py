"""Validate the actual PQ + full-RGB/limited-YUV NVENC path with known luminance patches."""
from pathlib import Path
import subprocess,json
import numpy as np
from color_pipeline import pq_encode,pq_decode
ROOT=Path(__file__).resolve().parents[1]
FFMPEG=next((ROOT/'dependencies/ffmpeg').rglob('ffmpeg.exe'))
levels=np.array([.005,1,10,50,100,200,500,1000],dtype=np.float32)
img=np.zeros((128,640,3),np.float32)
for k,level in enumerate(levels):img[:,k*80:(k+1)*80]=pq_encode(level)
raw=(img*65535+.5).astype('<u2').tobytes()
clip=ROOT/'intermediate/hdr_encoding_test.mp4'
cmd=[str(FFMPEG),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb48le','-s','640x128','-r','30','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt2020,format=p010le','-c:v','hevc_nvenc','-profile:v','main10','-preset','p7','-cq','16','-b:v','0','-color_range','tv','-colorspace','bt2020nc','-color_primaries','bt2020','-color_trc','smpte2084',str(clip)]
subprocess.run(cmd,input=raw*10,check=True,capture_output=True)
decode=[str(FFMPEG),'-v','error','-i',str(clip),'-frames:v','1','-vf','scale=in_range=limited:out_range=full:in_color_matrix=bt2020','-pix_fmt','rgb48le','-f','rawvideo','pipe:1']
data=subprocess.run(decode,check=True,capture_output=True).stdout
rgb=np.frombuffer(data,'<u2').reshape(128,640,3)/65535
measured=np.array([pq_decode(rgb[32:96,k*80+20:k*80+60]).mean() for k in range(8)])
error=np.abs(measured-levels)/np.maximum(levels,1)
report={'input_nits':levels.tolist(),'decoded_nits':measured.tolist(),'relative_error':error.tolist(),'pass':bool(np.all(error<.02)),'encode_command':cmd,'decode_command':decode}
(ROOT/'logs/hdr_encoding_test.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if 'command' not in k},indent=2))
assert report['pass'],'HDR encoding round trip exceeds 2%'
