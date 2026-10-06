"""Independent physical-height and motion stress fixtures, not road validation."""
from pathlib import Path
import sys,json,math,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prism.geometry import Camera,TTCEstimator,range_from_height
O=ROOT/'results/extended'
def main():
    cam=Camera(960,720,60);rows=[]
    conditions=['constant','noise','acceleration','dropout','truncation','identity_reset','receding','stationary']
    for cond in conditions:
        for H in [1.2,1.5,1.7,2.0]:
            for seed in range(50):
                rng=np.random.default_rng(20260914+seed);est=TTCEstimator();tid=1
                for i in range(100):
                    t=i/20;v=5.;a=1.5 if cond=='acceleration' else 0.
                    if cond=='receding':v=-2.
                    if cond=='stationary':v=0.
                    Z=50-v*t-.5*a*t*t;closing=v+a*t
                    if cond=='dropout' and rng.random()<.3:continue
                    if cond=='identity_reset' and i and i%10==0:tid+=1
                    h=cam.fx*H/Z
                    if cond=='truncation' and 30<=i<60:h*=.65
                    if cond in ['noise','receding','stationary']:h=max(1,h+rng.normal(0,2+.03*h))
                    est.update(tid,t,h);ttc,r2=est.ttc(tid);zhat,_=range_from_height('person',h,cam)
                    gt=Z/closing if closing>0 else None
                    rows.append(dict(condition=cond,height_m=H,seed=seed,frame=i,t=t,depth_true=Z,
                                     depth_est=zhat,depth_signed_pct=100*(zhat-Z)/Z,
                                     depth_ape_pct=100*abs(zhat-Z)/Z,ttc_true=gt,
                                     ttc_est=ttc if math.isfinite(ttc) else None,r2=r2,
                                     ttc_abs_error=abs(ttc-gt) if gt is not None and math.isfinite(ttc) else None))
    summary=[]
    for cond in conditions:
        for height in [None,1.2,1.5,1.7,2.0]:
            r=[x for x in rows if x['condition']==cond and (height is None or x['height_m']==height)]
            e=[x for x in r if x['ttc_true'] is not None];f=[x for x in e if x['ttc_est'] is not None];err=[x['ttc_abs_error'] for x in f]
            summary.append(dict(condition=cond,height_m=height,observations=len(r),
                depth_signed_median_pct=float(np.median([x['depth_signed_pct'] for x in r])),
                depth_median_ape_pct=float(np.median([x['depth_ape_pct'] for x in r])),
                ttc_eligible=len(e),ttc_finite=len(f),ttc_availability=len(f)/len(e) if e else None,
                ttc_median_abs_error=float(np.median(err)) if err else None,
                ttc_p95_abs_error=float(np.quantile(err,.95)) if err else None,
                false_closing_fraction=sum(x['ttc_est'] is not None for x in r)/len(r) if not e else None))
    import gzip
    with gzip.open(O/'geometry_rows.jsonl.gz','wt') as out:
        for row in rows:out.write(json.dumps(row,allow_nan=False)+'\n')
    (O/'geometry_summary.json').write_text(json.dumps(dict(seed_base=20260914,replicates=50,summary=summary,
        scope='Analytic stress fixtures. TTC reference is instantaneous Z/closing speed, not accelerated time to impact. Repeats in noiseless conditions are identical, not independent evidence.'),indent=2)+'\n')
    print(json.dumps([x for x in summary if x['height_m'] is None],indent=2))
if __name__=='__main__':main()
