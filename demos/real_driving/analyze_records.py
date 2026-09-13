"""Descriptive summaries of recorded real-video output; no ground-truth metrics."""
from pathlib import Path
import gzip,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 rows=[json.loads(x) for x in gzip.decompress((ROOT/'recorded/frame_records.jsonl.gz').read_bytes()).decode().splitlines()]
 assert len(rows)==1200
 policies={}
 for name,key in [('All tracked objects','tracks'),('Risk budget','risk_cues'),('PRISM','cues')]:
  prev=set();changes=0;counts=[]
  for r in rows:
   cur={t['tid'] for t in r[key]};counts.append(len(cur));changes+=len(cur^prev);prev=cur
  policies[name]={'mean_count':float(np.mean(counts)),'max_count':max(counts),'empty_frames':counts.count(0),'identity_additions_removals_per_s':changes/60}
 timing={}
 for name,keys in [('Detection and tracking',['perception_track_ms']),('State and scoring',['risk_ms']),('Cue allocation',['arbitration_ms']),('State scoring and allocation',['risk_ms','arbitration_ms']),('Pipeline excluding rendering',['perception_track_ms','risk_ms','arbitration_ms'])]:
  values=[sum(r['timings'][k] for k in keys) for r in rows[20:]]
  timing[name]={'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))}
 flagged=[]
 for i,r in enumerate(rows):
  for t in r['tracks']:
   x1,y1,x2,y2=t['box']
   if x2-x1>.75*1164 and y1>.55*874 and y2>.95*874:
    flagged.append({'frame':i,'t':r['t'],'tid':t['tid'],'confidence':t['conf'],'priority':t['risk'],'box':t['box'],'depth_m':t['Z'],'selected':any(q['tid']==t['tid'] for q in r['cues'])})
 result={'frames':len(rows),'policies':policies,'timing_frames':[20,1199],'timing':timing,'timing_boundary':'Recorded sequential CPU calls; excludes decode, rendering, encoding and display. First 20 frames omitted descriptively.','lower_image_candidate_audit':{'criterion':'width > 0.75W, top > 0.55H, bottom > 0.95H; post-hoc geometric screen, not a false-positive label set','candidate_observations':len(flagged),'selected_observations':sum(t['selected'] for t in flagged),'observations':flagged},'notes':'One selected sequence; no object, risk, event or driver ground truth. Counts are not recall, precision or safety metrics.'}
 (ROOT/'recorded/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='lower_image_candidate_audit'},indent=2))
if __name__=='__main__':main()
