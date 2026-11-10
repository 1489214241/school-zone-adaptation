"""Run beside all 14 .part files and parts-manifest.json to verify and restore ZIP."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'parts-manifest.json').read_text(encoding='utf-8'))
name = manifest['archive']
assert Path(name).name == name
output = root / name
if output.exists():
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest['sha256']
    print('Existing ZIP verified:', output)
else:
    temporary = root / (name + '.assembling')
    overall = hashlib.sha256()
    count = 0
    with temporary.open('xb') as destination:
        for part in manifest['parts']:
            assert Path(part['name']).name == part['name']
            data = (root / part['name']).read_bytes()
            assert len(data) == part['bytes'], part['name']
            assert hashlib.sha256(data).hexdigest() == part['sha256'], part['name']
            destination.write(data)
            overall.update(data)
            count += len(data)
    assert count == manifest['bytes']
    assert overall.hexdigest() == manifest['sha256']
    temporary.rename(output)
    print('All parts and complete ZIP verified:', output)
