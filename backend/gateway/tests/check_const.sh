#!/usr/bin/env bash
set -euo pipefail
P=~/secure-search/tests/result.json
python3 - <<'PY'
import json,base64,os,sys
p=os.path.expanduser('~/secure-search/tests/result.json')
d=json.load(open(p))
sc=d.get('scoresCiphertexts',[])
ps=d.get('packShapes',[])
lens={len(base64.b64decode(x)) for x in sc}
ok = (len(sc)==8) and (len(ps)==8) and (len(lens)==1 and list(lens)[0]==16384)
print("[check]", "OK" if ok else f"FAIL ct={len(sc)} ps={len(ps)} lens={lens}")
sys.exit(0 if ok else 1)
PY
