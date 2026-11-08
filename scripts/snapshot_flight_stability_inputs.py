"""Preserve and check the earlier pilot output file set without writing to it."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
source=root/'outputs/lamp_flight_residual_pilot'
target=root/'outputs/flight_residual_stability/previous_outputs_sha256.json'
if not source.is_dir():raise FileNotFoundError('Prior flight pilot output is required for diagnosis')
files=[{'path':str(p.relative_to(source)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
       for p in sorted(source.rglob('*')) if p.is_file() and not any(x.startswith('._') for x in p.relative_to(source).parts)]
if target.exists():
    if json.loads(target.read_text())!={'files':files}:raise RuntimeError('Earlier pilot outputs differ from the preserved snapshot')
else:
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps({'files':files},indent=2)+'\n')
print('Earlier pilot integrity snapshot:',len(files),'files')
