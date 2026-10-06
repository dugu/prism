"""Painted support and cue-overlap metrics on native-resolution black canvases."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from extended_study import load_sequences,evaluate,FAMILIES
from prism.study import make_policy
from prism.replay import RankedBudget
from prism.render import overlay_mask

def main():
    rows=[]
    for seq in load_sequences():
        # Reuse each candidate's mask between policy outputs on the same frame.
        policies={f:make_policy(f) for f in FAMILIES};policies['risk_budget']=RankedBudget('risk')
        shape=(seq['cam'].height,seq['cam'].width,3)
        for i,fr in enumerate(seq['frames']):
            cache={}
            for fam,pol in policies.items():
                sel=pol.step(fr['tracks'],seq['cam'].width,seq['cam'].height);acc=np.zeros(shape[:2],np.uint16)
                for cue in sel:
                    key=(cue['tid'],cue['level'])
                    if key not in cache:cache[key]=overlay_mask(shape,[cue],ribbon=False,panel=False)
                    acc+=cache[key]
                rows.append(dict(sequence=seq['name'],condition=seq['condition'],frame=i,family=fam,
                    cues=len(sel),surrogate_cost=sum(c['ink'] for c in sel),
                    painted_fraction=float(np.mean(acc>0)),overlap_fraction=float(np.mean(acc>1))))
        print(seq['name'],flush=True)
    summaries=[]
    for cond in ['all','none','night','fog','rain']:
        for fam in [*FAMILIES,'risk_budget']:
            r=[x for x in rows if x['family']==fam and (cond=='all' or x['condition']==cond)]
            sequences=sorted({x['sequence'] for x in r})
            means=[np.mean([x['painted_fraction'] for x in r if x['sequence']==s]) for s in sequences]
            summaries.append(dict(condition=cond,family=fam,sequence_mean_painted_pct=100*float(np.mean(means)),
                frame_p95_painted_pct=100*float(np.quantile([x['painted_fraction'] for x in r],.95)),
                sequence_mean_overlap_pct=100*float(np.mean([np.mean([x['overlap_fraction'] for x in r if x['sequence']==s]) for s in sequences])),
                frame_max_painted_pct=100*max(x['painted_fraction'] for x in r)))
    O=ROOT/'results/extended'
    (O/'occupancy_frames.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    (O/'occupancy_summary.json').write_text(json.dumps(dict(summary=summaries,numpy=np.__version__,
       definition='Union of nonzero per-cue masks rendered on black native-resolution canvases; panel/ribbon excluded. Pixel support, not perceived clutter.'),indent=2)+'\n')
if __name__=='__main__':main()
