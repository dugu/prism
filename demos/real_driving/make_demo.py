"""Run the unchanged PRISM pipeline on the official comma2k19 example video.

Outputs are a qualitative demonstration, not an annotated road-safety benchmark.
Run with .venv/bin/python make_demo.py [--replay] from this folder.
"""
from pathlib import Path
import argparse, ast, gzip, hashlib, json, math, os, platform, subprocess, sys, time
import cv2
import numpy as np
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from prism.geometry import Camera
from prism.pipeline import Pipeline
from prism.replay import RankedBudget
from prism.render import render_frame, band_colour
from prism.arbitration import ArbiterConfig

OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
BG = (24, 21, 18)
WHITE = (235, 240, 243)

def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    return x

def text(im, s, xy, scale=.6, col=WHITE, thickness=1):
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv2.LINE_AA)

def writer(name, width, height, fps):
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
        '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUT / name)], stdin=subprocess.PIPE)
    return proc

def tracked_image(frame, tracks):
    out = frame.copy()
    for tr in tracks:
        x1,y1,x2,y2 = map(int,tr['box'])
        cv2.rectangle(out,(x1,y1),(x2,y2),(230,205,100),2,cv2.LINE_AA)
        text(out, f"{tr['cls']} #{tr['tid']}",(max(2,x1),max(20,y1-6)),.52,(230,205,100))
    return out

def pane(frame, title, lines, cues=None):
    out = np.full((500,960,3),BG,np.uint8)
    text(out,title,(16,27),.70,thickness=2)
    ih = 452; iw = round(frame.shape[1]*ih/frame.shape[0])
    out[40:40+ih,8:8+iw] = cv2.resize(frame,(iw,ih),interpolation=cv2.INTER_AREA)
    x = iw+26
    for i,s in enumerate(lines): text(out,s,(x,67+i*30),.52)
    if cues:
        top = max(cues,key=lambda c:c['risk'])
        text(out,'Highest selected score',(x,265),.51)
        text(out,f"{top['cls']}  #{top['tid']}",(x,297),.67,band_colour(top['risk']),2)
        text(out,f"Priority: {top['risk']:.2f}",(x,332),.6)
        text(out,f"Detail: {top['level']}",(x,362),.55)
    return out

def main():
    args=argparse.ArgumentParser();args.add_argument('--replay',action='store_true');args=args.parse_args()
    times=np.load(ROOT/'source/frame_times.npy'); times=times-times[0]
    assert np.all(np.diff(times)>0)
    fps=20.0 # 0.05 s acquisition spacing; elementary HEVC reports a misleading 25 FPS.
    cap=cv2.VideoCapture(str(ROOT/'source/video.hevc'))
    w,h=int(cap.get(3)),int(cap.get(4))
    K=np.array(ast.literal_eval((ROOT/'source/camera_intrinsics.txt').read_text()))
    assert (w,h)==(1164,874) and K[0,0]==K[1,1] and K[0,2]==w/2 and K[1,2]==h/2
    cam=Camera(w,h,math.degrees(2*math.atan(w/(2*K[0,0]))))
    assert abs(cam.fx-K[0,0])<1e-8
    records=[]
    if args.replay:
        cache = OUT/'frame_records.jsonl'
        content = cache.read_text() if cache.exists() else gzip.decompress((ROOT/'recorded/frame_records.jsonl.gz').read_bytes()).decode()
        records=[json.loads(l) for l in content.splitlines()]
    else:
        from ultralytics import YOLO
        import torch, ultralytics
        torch.set_num_threads(2)
        pipeline=Pipeline(YOLO(str(ROOT/'source/yolo11n.pt')),cam,fps=fps,conf=.25,imgsz=640)
        baseline=RankedBudget('risk')
        env={'python':sys.version,'platform':platform.platform(),'ultralytics':ultralytics.__version__,
             'torch':torch.__version__,'opencv':cv2.__version__,'numpy':np.__version__,
             'torch_threads':torch.get_num_threads(),'device':'cpu','backend':'PyTorch FP32',
             'repository_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
        (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    single=writer('PRISM_real_driving.mp4',w,h+140,fps)
    compare=writer('PRISM_real_driving_comparison.mp4',1920,1080,fps)
    log=None if args.replay else (OUT/'frame_records.jsonl').open('w')
    start=time.perf_counter(); n=0
    try:
        for i,t in enumerate(times):
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Decode ended at frame {i} of {len(times)}')
            if args.replay: out=records[i]
            else:
                out=pipeline.step(frame,timestamp=float(t))
                out['risk_cues']=baseline.step(out['tracks'],w,h)
                out=clean(out);records.append(out)
                log.write(json.dumps(out,allow_nan=False)+'\n')
            tracks,cues,risk=out['tracks'],out['cues'],out['risk_cues']
            assert len(cues)<=5 and sum(c['ink'] for c in cues)<=.055+1e-12
            overlay=render_frame(frame,cues,panel=False,ribbon=False)
            rb=render_frame(frame,risk,panel=False,ribbon=False)
            alltracks=tracked_image(frame,tracks)
            show=np.full((h+140,w,3),BG,np.uint8)
            text(show,'PRISM | Real driving demonstration',(20,33),.85,thickness=2)
            text(show,f"Time {t:05.2f} s    Tracked objects {len(tracks)}    Selected cues {len(cues)}",(20,62),.62)
            show[80:80+h]=overlay
            text(show,'Actual algorithm output | YOLO11n + ByteTrack | Offline processing, 1x playback',(16,h+105),.51)
            text(show,'comma2k19 / comma.ai | Priority, range and TTC are heuristic estimates',(16,h+128),.50)
            comp=np.full((1080,1920,3),BG,np.uint8)
            text(comp,f'PRISM | Same real frames and tracked objects | {t:05.2f} s',(18,32),.83,thickness=2)
            panels=[pane(frame,'1  ORIGINAL CAMERA', ['Real recorded video','No synthetic objects','Full camera image', 'Playback: 1x / 20 FPS']),
                    pane(alltracks,'2  ALL TRACKED OBJECTS', [f'Objects: {len(tracks)}','YOLO11n + ByteTrack','Supported road classes','No cue selection']),
                    pane(rb,'3  RISK-RANKED BUDGET', [f'Cues: {len(risk)} / {len(tracks)}','Ranked by risk score','Same cue types / budget','No persistence'],risk),
                    pane(overlay,'4  PRISM', [f'Cues: {len(cues)} / {len(tracks)}','Priority + persistence','Detail / budget control','Unchanged policy'],cues)]
            for j,p in enumerate(panels):comp[48+(j//2)*500:548+(j//2)*500,(j%2)*960:(j%2+1)*960]=p
            text(comp,'Source: comma2k19 / comma.ai | Offline qualitative demonstration; scores are not collision probabilities.',(18,1071),.58)
            single.stdin.write(show.tobytes());compare.stdin.write(comp.tobytes())
            if i in (0,100,200,400,600,800,1000,1199):
                cv2.imwrite(str(OUT/f'comparison_{i:04d}.jpg'),comp)
                cv2.imwrite(str(OUT/f'prism_{i:04d}.jpg'),show)
            n+=1
            if n%100==0:print(f'{n}/{len(times)} frames | {time.perf_counter()-start:.1f} s wall time | {len(tracks)} tracks / {len(cues)} cues',flush=True)
        assert not cap.read()[0], 'Extra video frames not represented in timestamps'
    finally:
        cap.release()
        if log:log.close()
        for proc in (single,compare):
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    stats={'frames':n,'playback_duration_s':n/fps,'acquisition_duration_s':float(times[-1]),
        'timestamp_spacing_s':dict(zip(['min','median','max'],map(float,np.quantile(np.diff(times),[0,.5,1])))),
        'camera':{'width':w,'height':h,'fx':cam.fx,'hfov_deg':cam.hfov_deg},
        'config':{'model':'yolo11n.pt','imgsz':640,'confidence':.25,'fps':fps,'arbiter':vars(ArbiterConfig())},
        'mean_tracks':float(np.mean([len(r['tracks']) for r in records])),
        'mean_prism_cues':float(np.mean([len(r['cues']) for r in records])),
        'mean_risk_cues':float(np.mean([len(r['risk_cues']) for r in records])),
        'prism_zero_cue_frames':sum(not r['cues'] for r in records),
        'prism_max_cues':max(len(r['cues']) for r in records),
        'prism_max_surrogate_cost':max(sum(c['ink'] for c in r['cues']) for r in records),
        'processing_note':'Offline PyTorch CPU run, not a reproduction of manuscript backend latency. No ground-truth validation.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').iterdir() if p.is_file()}}
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
