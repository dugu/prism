"""Paired geometry-to-selection diagnostic using the archived analytic trajectories.

Only the focal candidate's Z and TTC differ between branches. Candidate boxes,
identities, confidence, age, quality index and competitors are identical. This
isolates the causal effect of geometry on heuristic selection, not road validity.
"""
from pathlib import Path
import sys,json,gzip,math,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prism.risk import TrackState,risk_score
from prism.arbitration import Arbiter,ArbiterConfig
OUT=ROOT/'results/revision_20261006';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'results/extended/geometry_rows.jsonl.gz'
def candidate(tid,z,ttc,box,age=10):
 s=TrackState(tid,'person',.9,box,z,0.,0.,ttc,1.,age)
 return dict(tid=tid,box=box,risk=risk_score(s)['risk'],uncertainty=.2)
def run():
 groups=collections.defaultdict(list)
 with gzip.open(SOURCE,'rt') as f:
  for line in f:
   x=json.loads(line);groups[x['condition'],x['height_m'],x['seed']].append(x)
 records=[]
 for (condition,height,seed),rows in groups.items():
  for k in (1,5):
   policies=[Arbiter(ArbiterConfig(max_cues=k)) for _ in range(2)]
   for x in rows:
    # Fixed disjoint boxes avoid conflating geometry scoring with overlap rejection.
    competitors=[candidate(i+2,z,ttc,(20+180*i,100,60+180*i,180))
                 for i,(z,ttc) in enumerate([(12,2.5),(20,4.),(28,5.5),(36,float('inf')),(44,float('inf'))])]
    true= candidate(1,x['depth_true'],x['ttc_true'] if x['ttc_true'] is not None else float('inf'),(450,400,490,480))
    est= candidate(1,x['depth_est'],x['ttc_est'] if x['ttc_est'] is not None else float('inf'),true['box'])
    pools=[[true,*competitors],[est,*competitors]]
    selections=[{t['tid'] for t in a.step(pool,960,720)} for a,pool in zip(policies,pools)]
    ranks=[sorted(pool,key=lambda q:(-q['risk'],q['tid'])) for pool in pools]
    records.append(dict(condition=condition,height=height,seed=seed,frame=x['frame'],K=k,
     true_risk=true['risk'],estimated_risk=est['risk'],ttc_error=x['ttc_abs_error'],
     rank_changed=[q['tid'] for q in ranks[0]] != [q['tid'] for q in ranks[1]],
     set_changed=selections[0]!=selections[1],focal_true=1 in selections[0],focal_est=1 in selections[1],
     extra=1 not in selections[0] and 1 in selections[1],omitted=1 in selections[0] and 1 not in selections[1]))
 summary=[]
 for cond in sorted({r['condition'] for r in records}):
  for k in (1,5):
   rr=[r for r in records if r['condition']==cond and r['K']==k]
   large=[r for r in rr if r['ttc_error'] is not None and r['ttc_error']>10]
   summary.append(dict(condition=cond,K=k,n=len(rr),**{key:sum(r[key] for r in rr) for key in ['rank_changed','set_changed','extra','omitted','focal_true','focal_est']},
      large_ttc_error_n=len(large),large_error_set_changed=sum(r['set_changed'] for r in large)))
 with gzip.open(OUT/'selection_stress_rows.jsonl.gz','wt') as f:
  for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
 payload=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),summary=summary,
   controls='Same six candidates in both branches: focal geometry true vs estimated; five fixed competitors. Confidence .9, age 10, X 0, quality .2, disjoint fixed boxes, default B .055, K 1 or 5. Dropout skips the focal trajectory sample in both branches; identity-reset effects enter through archived TTC only. No claim to measure full perception or identity-loss effects. Noiseless replicates repeat identical trajectories.')
 (OUT/'selection_stress_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 for r in summary:print(r['condition'],r['K'],r['n'],*[round(100*r[f]/r['n'],2) for f in ['rank_changed','set_changed','extra','omitted']],r['large_ttc_error_n'],r['large_error_set_changed'])
 # Causal controls: restoring exact geometry must restore all scores and decisions.
 for k in (1,5):
  aa,bb=Arbiter(ArbiterConfig(max_cues=k)),Arbiter(ArbiterConfig(max_cues=k))
  for n in range(100):
   c=[candidate(1,50-n*.25,(50-n*.25)/5,(450,400,490,480))]
   assert aa.step(c,960,720)==bb.step(c,960,720)
 assert all(r['extra']+r['omitted']<=r['set_changed'] for r in summary)
 print('Paired controls and count consistency passed.')
if __name__=='__main__':run()
