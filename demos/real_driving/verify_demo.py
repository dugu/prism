"""Verify committed decisions and optionally decode the two local videos."""
from pathlib import Path
import argparse,gzip,json,math,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from prism.geometry import Camera
from prism.replay import StateReplay,RankedBudget
from prism.arbitration import Arbiter

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return None if isinstance(x,float) and not math.isfinite(x) else x

def equivalent(a,b):
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
 if isinstance(a,float) and isinstance(b,(float,int)):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
 return a==b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--decisions-only',action='store_true');args=ap.parse_args()
 s=json.loads((ROOT/'recorded/summary.json').read_text());c=s['camera'];cam=Camera(c['width'],c['height'],c['hfov_deg'])
 state=StateReplay(cam);arb=Arbiter();baseline=RankedBudget('risk')
 rows=[json.loads(l) for l in gzip.decompress((ROOT/'recorded/frame_records.jsonl.gz').read_bytes()).decode().splitlines()]
 assert len(rows)==1200==s['frames']
 for row in rows:
  raw=[{k:t[k] for k in ('tid','cls','conf','box','age')} for t in row['tracks']]
  tracks=state.step(raw,row['t']);assert equivalent(clean(tracks),row['tracks'])
  assert equivalent(clean(arb.step(tracks,cam.width,cam.height)),row['cues'])
  assert equivalent(clean(baseline.step(tracks,cam.width,cam.height)),row['risk_cues'])
 videos={}
 if not args.decisions_only:
  import cv2
  for name in ('PRISM_real_driving.mp4','PRISM_real_driving_comparison.mp4'):
   cap=cv2.VideoCapture(str(ROOT/'output'/name));fps=cap.get(cv2.CAP_PROP_FPS);dimensions=[int(cap.get(3)),int(cap.get(4))];n=0
   while True:
    ok,frame=cap.read()
    if not ok:break
    assert frame.shape[:2]==(dimensions[1],dimensions[0]);n+=1
   cap.release();assert n==len(rows) and abs(fps-20)<1e-6
   videos[name]={'decoded_frames':n,'fps':fps,'dimensions':dimensions,'duration_s':n/fps}
 print(json.dumps({'replayed_frames':len(rows),'all_states_and_policy_decisions_match':True,'videos':videos},indent=2))
if __name__=='__main__':main()
