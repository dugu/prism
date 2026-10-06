"""Fixed-schedule diagnostic contact sheets for visible ego-vehicle candidates.

Requires downloaded original video; does not establish detection-wide accuracy.
"""
from pathlib import Path
import sys,json,gzip,argparse
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'demos/real_driving';O=ROOT/'results/extended'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sheets',type=Path,required=True);args=ap.parse_args();args.sheets.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in gzip.decompress((D/'recorded/frame_records.jsonl.gz').read_bytes()).decode().splitlines()]
    cap=cv2.VideoCapture(str(D/'source/video.hevc'));sample=[];tiles=[]
    for i in range(1200):
        ok,frame=cap.read()
        if not ok:raise RuntimeError(i)
        if i%20:continue
        r=rows[i];H,W=frame.shape[:2];candidates=[];selected={x['tid'] for x in r['cues']}
        for t in r['tracks']:
            x1,y1,x2,y2=t['box']
            if x2-x1>.75*W and y1>.55*H and y2>.95*H:
                candidates.append(dict(tid=t['tid'],box=t['box'],cls=t['cls'],conf=t['conf'],selected=t['tid'] in selected))
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),3)
        tile=cv2.resize(frame,(582,437));cv2.rectangle(tile,(0,0),(582,29),(0,0,0),-1)
        cv2.putText(tile,f'frame {i} | {i/20:.0f}s | candidates {len(candidates)}',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile);sample.append(dict(frame=i,time_s=r['t'],screened_candidates=candidates))
    cap.release()
    for j in range(5):
        panel=np.vstack([np.hstack(tiles[j*12+k:j*12+k+3]) for k in range(0,12,3)])
        cv2.imwrite(str(args.sheets/f'audit-{j+1}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,94])
    (O/'video_audit_sample.json').write_text(json.dumps(dict(schedule='Frames 0,20,...,1180, one frame per second. Screen: width>.75W, top>.55H, bottom>.95H. Narrow ego-region diagnostic only.',sample=sample),indent=2)+'\n')
    print('samples',len(sample),'candidate observations',sum(len(x['screened_candidates']) for x in sample))
if __name__=='__main__':main()
