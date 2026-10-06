"""Run the protocol in results/extended/protocol.md; no detector inference."""
from pathlib import Path
import sys,json,itertools,collections,platform,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prism.geometry import Camera
from prism.replay import StateReplay,RankedBudget
from prism.arbitration import Arbiter,ArbiterConfig
from prism.study import make_policy,temporal_metrics
O=ROOT/'results/extended';O.mkdir(exist_ok=True)
FAMILIES=['threshold','hysteresis','dwell','PRISM']
def save(name,x):
    (O/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mean(x):return float(np.mean(x)) if len(x) else None

def load_sequences():
    out=[]
    for p in sorted((ROOT/'data/experiment/results/tracks').glob('*.json')):
        d=json.loads(p.read_text());cam=Camera(d['meta']['width'],d['meta']['height'],d['meta']['hfov_deg']);state=StateReplay(cam)
        frames=[{**f,'tracks':state.step(f['tracks'],f['t'])} for f in d['frames']]
        out.append(dict(name=p.stem,scene=p.stem.split('_')[0],condition=p.stem.split('_')[1],cam=cam,fps=d['fps'],frames=frames))
    return out

def evaluate(seq,policy,trace=False):
    prev=set();H=D=S=changes=below=total=0;counts=[];costs=[];records=[];selections=[]
    for f in seq['frames']:
        sel=policy.step(f['tracks'],seq['cam'].width,seq['cam'].height);ids={t['tid'] for t in sel}
        eligible=f['prim_risk_gt']>=.35;hit=eligible and f['prim_tid'] in ids
        H+=eligible;D+=eligible and f['prim_tid'] is not None;S+=hit
        changes+=len(ids^prev);prev=ids;counts.append(len(sel));costs.append(sum(x['ink'] for x in sel))
        below+=sum(x['risk']<.24 for x in sel);total+=len(sel)
        records.append(dict(eligible=eligible,reference_tid=f['prim_tid'],hit=bool(hit)))
        if trace:selections.append(sel)
    result=dict(sequence=seq['name'],scene=seq['scene'],condition=seq['condition'],H=H,D=D,S=S,
                coverage=S/H if H else None,cues=mean(counts),cost=mean(costs),
                transitions_per_s=changes/(len(counts)/seq['fps']),below_activation_cue_frames=below,total_cue_frames=total,
                **temporal_metrics(records,seq['fps']))
    return (result,selections,records) if trace else result

def aggregate(rows):
    H=sum(r['H'] for r in rows);D=sum(r['D'] for r in rows);S=sum(r['S'] for r in rows)
    delays=[v for r in rows for v in r['conditional_first_display_delays_s']]
    total=sum(r['total_cue_frames'] for r in rows)
    return dict(H=H,D=D,S=S,coverage=S/H if H else None,conditional=S/D if D else None,
                cues=mean([r['cues'] for r in rows]),cost=mean([r['cost'] for r in rows]),
                transitions_per_s=mean([r['transitions_per_s'] for r in rows]),
                longest_eligible_omission_s=max(r['longest_eligible_omission_s'] for r in rows),
                bouts=sum(r['bouts'] for r in rows),never_displayed_bouts=sum(r['never_displayed_bouts'] for r in rows),
                displayed_bout_delay_median_s=float(np.median(delays)) if delays else None,
                displayed_bout_delay_p95_s=float(np.quantile(delays,.95)) if delays else None,
                below_activation_fraction=sum(r['below_activation_cue_frames'] for r in rows)/total if total else 0)

def main():
    seqs=load_sequences();grid=[];default=[];traces=[]
    for fam,on,budget,k in itertools.product(FAMILIES,[0,.12,.18,.24,.30,.36,.42,.48,.54,.60],[.0165,.033,.055],[1,3,5]):
        rows=[evaluate(s,make_policy(fam,on,budget,k)) for s in seqs]
        grid.append(dict(family=fam,on=on,budget=budget,count=k,rows=rows))
    save('grid.json',grid);print('grid complete: 360 configurations x 32 sequences',flush=True)
    for fam in [*FAMILIES,'risk_budget']:
        for s in seqs:
            policy=RankedBudget('risk') if fam=='risk_budget' else make_policy(fam)
            r,selections,records=evaluate(s,policy,True);default.append(dict(family=fam,**r))
            traces.extend(dict(sequence=s['name'],frame=i,t=s['frames'][i]['t'],family=fam,
                               selected_ids=[x['tid'] for x in sel],selected_levels=[x['level'] for x in sel],
                               cues=len(sel),**rec) for i,(sel,rec) in enumerate(zip(selections,records)))
    save('default_sequences.json',default);save('default_traces.json',traces)
    defaults={cond:{fam:aggregate([r for r in default if r['family']==fam and (cond=='all' or r['condition']==cond)]) for fam in [*FAMILIES,'risk_budget']} for cond in ['all','none','night','fog','rain']}
    folds=[];cv=[]
    for lam in [0,.025,.05,.10]:
        for fam in FAMILIES:
            chosen=[]
            for held in sorted({s['scene'] for s in seqs}):
                options=[g for g in grid if g['family']==fam]
                def utility(g):
                    r=[x for x in g['rows'] if x['scene']!=held]
                    return mean([x['coverage'] for x in r if x['coverage'] is not None])-lam*mean([x['cues'] for x in r])-.005*mean([x['transitions_per_s'] for x in r])
                best=max(options,key=utility);test=[x for x in best['rows'] if x['scene']==held];chosen+=test
                folds.append(dict(lam=lam,family=fam,held_scene=held,on=best['on'],budget=best['budget'],count=best['count'],training_utility=utility(best),test_rows=test))
            cv.append(dict(lam=lam,family=fam,**aggregate(chosen)))
    save('folds.json',folds)
    boot=[]
    for lam in [0,.025,.05,.10]:
        for other in ['threshold','hysteresis','dwell']:
            diffs=[]
            for scene in sorted({s['scene'] for s in seqs}):
                a=next(x for x in folds if x['lam']==lam and x['family']=='PRISM' and x['held_scene']==scene)['test_rows']
                b=next(x for x in folds if x['lam']==lam and x['family']==other and x['held_scene']==scene)['test_rows']
                aa=aggregate(a);bb=aggregate(b)
                if aa['coverage'] is not None:diffs.append(aa['coverage']-bb['coverage'])
            rng=np.random.default_rng(20260914);v=np.array(diffs);bs=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1)
            boot.append(dict(lam=lam,comparator=other,eligible_scenes=len(v),mean_difference=float(v.mean()),ci95=np.quantile(bs,[.025,.975]).tolist()))
    # Default factorial and one-factor perturbations, never selected on outcomes.
    sensitivity=[]
    configs=[(f'factorial_h{h}_c{c}_q{q}',dict(use_hysteresis=bool(h),use_foveal=bool(c),use_uncertainty_gate=bool(q))) for h,c,q in itertools.product([0,1],repeat=3)]
    for field,values in [('dwell_frames',[0,4,8,16]),('hysteresis_bonus',[0,.03,.06,.12]),('foveal_penalty',[.25,.45,.75,1]),('u_gate',[.5,.6,.72,.9])]:
        configs.extend((f'{field}_{v}',{field:v}) for v in values)
    for label,kw in configs:
        rows=[evaluate(s,Arbiter(ArbiterConfig(**kw))) for s in seqs]
        sensitivity.append(dict(label=label,config=kw,all=aggregate(rows),clear=aggregate([r for r in rows if r['condition']=='none'])))
    save('sensitivity.json',sensitivity)
    save('summary.json',dict(defaults=defaults,scene_grouped_selection=cv,paired_bootstrap=boot,
         interpretation='Exploratory reuse; score-defined reference, not independently labelled hazard ground truth.',
         protocol_sha256=hashlib.sha256((O/'protocol.md').read_bytes()).hexdigest(),
         environment=dict(python=platform.python_version(),numpy=np.__version__)))
    print(json.dumps(dict(clear=defaults['none'],crossvalidation=cv),indent=2))
if __name__=='__main__':main()
