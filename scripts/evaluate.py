"""Reproduce all final-paper policy and state statistics without detector inference."""
from pathlib import Path
import sys, json, math, statistics, collections, platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prism.geometry import Camera
from prism.replay import StateReplay, policy_set
B=ROOT/'data/experiment/results';O=ROOT/'results';O.mkdir(exist_ok=True)

def finite(x):return x is not None and math.isfinite(x)
def avg(x):return float(np.mean(x)) if x else None
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(name,data): (O/name).write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')

def main():
    scene_rows=[];lookup={};frame_records=[];figure_cues={};invariants={'frames':0,'cue_limit_violations':0,'cost_violations':0}
    for f in sorted((B/'tracks').glob('*.json')):
        d=json.loads(f.read_text());cam=Camera(d['meta']['width'],d['meta']['height'],d['meta']['hfov_deg'])
        state=StateReplay(cam);policies=policy_set();names=[*policies,'Render all','Confidence matched count','Risk matched count']
        acc={k:dict(H=0,D=0,S=0,cues=[],cost=[],transitions=0,previous=set()) for k in names}
        for index,fr in enumerate(d['frames']):
            current=state.step(fr['tracks'],fr['t'])
            for old,new in zip(fr['tracks'],current):
                key=(f.stem,fr['t'],old['conf'],old['Z'])
                assert key not in lookup, ('ambiguous matched-state key',key)
                lookup[key]=new
            selected={name:p.step(current,cam.width,cam.height) for name,p in policies.items()}
            selected['Render all']=[{**tr,'level':'full','ink':.0165} for tr in current]
            n=len(selected['PRISM'])
            for label,key in [('Confidence matched count','conf'),('Risk matched count','risk')]:
                selected[label]=sorted(current,key=lambda x:(-x[key],x['tid']))[:n]
            H=fr['prim_risk_gt']>=.35;D=fr['prim_tid'] is not None
            record=dict(sequence=f.stem,frame=index,t=fr['t'],reference_eligible=H,reference_detected=D,policies={})
            for name,sel in selected.items():
                ids={x['tid'] for x in sel};cost=sum(x.get('ink',0) for x in sel);hit=H and fr['prim_tid'] in ids;a=acc[name]
                a['H']+=H;a['D']+=H and D;a['S']+=hit;a['cues'].append(len(sel));a['cost'].append(cost)
                a['transitions']+=len(ids^a['previous']);a['previous']=ids
                record['policies'][name]=dict(ids=sorted(ids),cues=len(sel),displayed_reference=bool(hit),cost=cost if name in policies else None)
                if name in policies:
                    invariants['cue_limit_violations']+=len(sel)>5
                    invariants['cost_violations']+=cost>.055+1e-12
            if f.stem in ['S1_none','S3_none'] and index in [20,40]:
                figure_cues[f.stem+'_'+str(index)]={k:clean(selected[k]) for k in ['PRISM','Render all','Confidence budget']}
            invariants['frames']+=1;frame_records.append(record)
        for name,a in acc.items():
            scene_rows.append(dict(scene=f.stem.split('_')[0],condition=f.stem.split('_')[1],policy=name,
                                   frames=len(d['frames']),H=a['H'],D=a['D'],S=a['S'],
                                   cues=avg(a['cues']),cost=avg(a['cost']) if name in policies else None,
                                   transitions_per_s=a['transitions']/(len(d['frames'])/d['fps']),
                                   conditional=a['S']/a['D'] if a['D'] else None,
                                   end_to_end=a['S']/a['H'] if a['H'] else None))
    aggregated=[]
    for condition in ['none','night','fog','rain']:
        for name in names:
            rows=[x for x in scene_rows if x['condition']==condition and x['policy']==name]
            H=sum(x['H'] for x in rows);D=sum(x['D'] for x in rows);S=sum(x['S'] for x in rows)
            aggregated.append(dict(condition=condition,policy=name,H=H,D=D,S=S,cues=avg([x['cues'] for x in rows]),
                                   cost=avg([x['cost'] for x in rows if x['cost'] is not None]),
                                   transitions_per_s=avg([x['transitions_per_s'] for x in rows]),
                                   conditional=S/D,end_to_end=S/H))
    matched=[]
    for row in json.loads((B/'e3_state_rows.json').read_text()):
        key=(row['scenario']+'_'+row['condition'],row['t'],row['conf'],row['Z_est'])
        assert key in lookup, ('missing state',key)
        tr=lookup[key]
        matched.append(dict(scene=row['scenario'],condition=row['condition'],t=row['t'],
                            Z_gt=row['Z_gt'],range_ape=abs(tr['Z']-row['Z_gt'])/row['Z_gt'],
                            ttc_gt=row['ttc_gt'],ttc_est=tr['ttc'],iou=row['iou']))
    state_stats=[]
    for cond in ['all','none','night','fog','rain']:
        m=[r for r in matched if cond=='all' or r['condition']==cond]
        eligible=[r for r in m if finite(r['ttc_gt']) and 0<r['ttc_gt']<=4]
        valid=[r for r in eligible if finite(r['ttc_est'])]
        state_stats.append(dict(condition=cond,matched=len(m),range_median_ape_pct=100*statistics.median(r['range_ape'] for r in m),
                                ttc_eligible=len(eligible),ttc_finite=len(valid),
                                ttc_availability=len(valid)/len(eligible),
                                ttc_median_abs_error=statistics.median(abs(r['ttc_est']-r['ttc_gt']) for r in valid)))
    boot=[]
    for comparator in ['Risk budget','Confidence budget','Risk matched count']:
        a=sorted([r for r in scene_rows if r['condition']=='none' and r['policy']=='PRISM'],key=lambda r:r['scene'])
        b=sorted([r for r in scene_rows if r['condition']=='none' and r['policy']==comparator],key=lambda r:r['scene'])
        diffs=np.array([x['end_to_end']-y['end_to_end'] for x,y in zip(a,b) if x['end_to_end'] is not None and y['end_to_end'] is not None])
        rng=np.random.default_rng(20260911);bs=diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))].mean(axis=1)
        boot.append(dict(comparator=comparator,eligible_scenes=len(diffs),mean_difference=float(diffs.mean()),
                         ci95=list(np.quantile(bs,[.025,.975])),seed=20260911,resamples=10000))
    assert invariants['cost_violations']==0 and invariants['cue_limit_violations']==0
    summary=dict(environment=dict(python=platform.python_version(),numpy=np.__version__),
                 invariants=invariants,policy_aggregates=aggregated,state_statistics=state_stats,paired_bootstrap=boot)
    save('final_summary.json',summary);save('scene_metrics.json',scene_rows);save('frame_metrics.json',frame_records)
    save('matched_states.json',matched);save('figure_cues.json',figure_cues)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
