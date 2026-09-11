"""Compare deterministic numerical results with the frozen reference summary."""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[1]
actual=json.loads((ROOT/'results/final_summary.json').read_text())
expected=json.loads((ROOT/'results/reference_summary.json').read_text())
def compare(a,b,path=''):
    if isinstance(b,dict):
        assert set(a)==set(b),path
        for k,v in b.items():
            if k!='environment':compare(a[k],v,path+'/'+k)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,(aa,bb) in enumerate(zip(a,b)):compare(aa,bb,path+'/'+str(i))
    elif isinstance(b,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(path,a,b)
    else:assert a==b,(path,a,b)
compare(actual,expected)
print('All final numerical summaries match the frozen reference within 1e-10 tolerance.')
