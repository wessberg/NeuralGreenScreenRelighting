"""Local reusable audio envelope/onset analysis for future music-driven scene cues.

These are measured audio features, not artist-approved lighting cues or a claimed
beat/downbeat grid. Output timing is in the supplied audio file's timeline.
"""
from pathlib import Path
import argparse,subprocess,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FFMPEG=next((ROOT/'dependencies/ffmpeg').rglob('ffmpeg.exe'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--duration',type=float);args=p.parse_args()
    cmd=[str(FFMPEG),'-v','error','-i',str(Path(args.audio).resolve())]
    if args.duration:cmd+=['-t',str(args.duration)]
    cmd+=['-vn','-ac','1','-ar','24000','-f','f32le','pipe:1']
    data=subprocess.run(cmd,capture_output=True,check=True).stdout
    audio=np.frombuffer(data,np.float32);sr=24000;hop=240;window=2048
    padded=np.pad(audio,(window//2,window//2));starts=np.arange(0,len(audio),hop)
    frames=np.stack([padded[k:k+window] for k in starts]);win=np.hanning(window)
    amplitude=np.sqrt(np.mean(frames*frames,axis=1));spectrum=np.abs(np.fft.rfft(frames*win,axis=1));freq=np.fft.rfftfreq(window,1/sr)
    flux=np.r_[0,np.maximum(np.diff(np.log1p(spectrum),axis=0),0).mean(1)]
    def normalize(v):return np.clip(v/max(float(np.percentile(v,98)),1e-9),0,1)
    envelope=normalize(amplitude);onset_strength=normalize(flux)
    bands={name:normalize(np.sqrt(np.mean(spectrum[:,(freq>=low)&(freq<high)]**2,axis=1))) for name,low,high in [('bass',40,250),('mid',250,2000),('high',2000,10000)]}
    peaks=[];last=-1000
    for i in range(2,len(flux)-2):
        if onset_strength[i]>.42 and flux[i]==max(flux[i-2:i+3]) and i-last>=12:
            peaks.append({'time_seconds':round(i*.01,3),'strength':round(float(onset_strength[i]),4)});last=i
    result={'source':str(Path(args.audio).resolve()),'analyzed_duration_seconds':len(audio)/sr,'analysis_sample_rate':sr,'feature_step_seconds':.01,'time_origin_seconds':0,'status':'Audio features only; no lighting animation applied. Onsets are not a validated beat grid.','onset_candidates':peaks,'features':[{'time_seconds':round(i*.01,3),'rms_envelope':round(float(envelope[i]),5),'onset_strength':round(float(onset_strength[i]),5),**{name:round(float(values[i]),5) for name,values in bands.items()}} for i in range(len(starts))]}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print('AUDIO_ANALYSIS',str(out.resolve()),'duration',len(audio)/sr,'onset_candidates',len(peaks))

if __name__=='__main__':main()
