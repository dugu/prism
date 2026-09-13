"""Download pinned public inputs and verify the recorded SHA-256 digests."""
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent

def main():
 meta=json.loads((ROOT/'source/source.json').read_text());sums=json.loads((ROOT/'recorded/summary.json').read_text())['source_sha256']
 urls={'video.hevc':f"https://raw.githubusercontent.com/commaai/comma2k19/{meta['commit']}/"+quote(meta['files']['video.hevc'],safe='/'),'yolo11n.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'}
 for name,url in urls.items():
  target=ROOT/'source'/name
  if not target.exists():
   temporary=target.with_suffix(target.suffix+'.part')
   with urlopen(url,timeout=90) as response,temporary.open('wb') as f:shutil.copyfileobj(response,f)
   assert hashlib.sha256(temporary.read_bytes()).hexdigest()==sums[name],name
   temporary.replace(target)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==sums[name],name
  print(name,'SHA-256 verified')
if __name__=='__main__':main()
