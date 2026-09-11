"""Check immutable input hashes, counts, export hashes and printed run records."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'data/experiment'
manifest=json.loads((ROOT/'data/input_sha256.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
assert len((B/'dataset/image_list.txt').read_text().splitlines())==554
assert sum(len(p.read_text().splitlines()) for p in (B/'dataset/labels').glob('*.txt'))==5644
assert len(json.loads((B/'results/e3_state_rows.json').read_text()))==12222
hashes=0
for line in (B/'models/sha256sums.txt').read_text().splitlines():
    fields=line.split(maxsplit=1)
    if len(fields)!=2:continue
    expected,name=fields;p=B/'models'/name
    if p.is_file():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
        hashes+=1
for log,result in [('e1_detector_benchmark','e1_detector_benchmark'),('e2_edge_backends','e2_backends')]:
    printed=[]
    for line in (B/'logs'/(log+'.log')).read_text().splitlines():
        if '{' not in line:continue
        try:value=ast.literal_eval(line[line.index('{'):])
        except (ValueError,SyntaxError):continue
        if isinstance(value,dict) and 'model' in value:printed.append(value)
    rows=json.loads((B/'results'/(result+'.json')).read_text())
    for row in rows:
        records=[p for p in printed if all(p.get(k)==row.get(k) for k in ['model','imgsz','backend'])]
        assert len(records)==1
        assert all(row.get(k)==v for k,v in records[0].items()),(log,row['model'],row['imgsz'])
    print(log,len(rows),'run records: all console-emitted fields agree')
print(len(manifest),'input file hashes passed;',hashes,'supplied model export hashes passed')
