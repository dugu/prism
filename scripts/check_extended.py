"""Check archived results against recomputed default decisions and geometry facts."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from extended_study import load_sequences,evaluate,aggregate,FAMILIES
from prism.study import make_policy
from prism.replay import RankedBudget
from prism.arbitration import Arbiter
R=ROOT/'results/extended'
def near(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(near(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(near(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10)
    return a==b

def main():
    summary=json.loads((R/'summary.json').read_text());rows=[]
    for seq in load_sequences():
        old=Arbiter();new=make_policy('PRISM')
        for frame in seq['frames']:
            a=old.step(frame['tracks'],seq['cam'].width,seq['cam'].height);b=new.step(frame['tracks'],seq['cam'].width,seq['cam'].height)
            assert [(t['tid'],t['level']) for t in a]==[(t['tid'],t['level']) for t in b]
        for fam in [*FAMILIES,'risk_budget']:
            policy=RankedBudget('risk') if fam=='risk_budget' else make_policy(fam)
            rows.append(dict(family=fam,**evaluate(seq,policy)))
    for cond,expected in summary['defaults'].items():
        for fam,v in expected.items():assert near(aggregate([r for r in rows if r['family']==fam and (cond=='all' or r['condition']==cond)]),v),(cond,fam)
    grid=json.loads((R/'grid.json').read_text());assert len(grid)==360
    for g in grid:
        assert len(g['rows'])==32
        assert all(r['cues']<=g['count'] and r['cost']<=g['budget']+1e-10 for r in g['rows'])
    folds=json.loads((R/'folds.json').read_text());assert len(folds)==128
    for f in folds:assert all(r['scene']==f['held_scene'] for r in f['test_rows'])
    g=json.loads((R/'geometry_summary.json').read_text())['summary']
    for h in [1.2,1.5,1.7,2.0]:
        row=next(x for x in g if x['condition']=='constant' and x['height_m']==h)
        assert abs(row['depth_signed_median_pct']-100*(1.7/h-1))<1e-10
        assert row['ttc_p95_abs_error']<1e-10
    print('Extended checks passed: unchanged PRISM identities/details; five replay families; grid capacities; scene grouping; analytic height/TTC anchors.')
if __name__=='__main__':main()
