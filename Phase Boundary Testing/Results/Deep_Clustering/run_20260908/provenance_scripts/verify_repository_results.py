from pathlib import Path
import hashlib
import json
import subprocess
import shutil

w=Path('\\\\?\\'+str(Path(__file__).resolve().parents[1]))
r=w/'work/CMU-vPCF-Project'
t=r/'Phase Boundary Testing/Results/Deep_Clustering/run_20260908'
shutil.copy2(Path(__file__),t/'provenance_scripts/verify_repository_results.py')
(t/'.gitattributes').write_text('# Preserve exact artifact bytes and SHA-256 hashes.\n* -text\n',encoding='utf-8')
files=sorted(p for p in t.rglob('*') if p.is_file() and p.name!='artifact_inventory.json')
inventory=[dict(path=p.relative_to(t).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(t/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
subprocess.run(['git','add','-f','--','Phase Boundary Testing/Results/Deep_Clustering/run_20260908'],cwd=r,check=True)
subprocess.run(['git','add','--renormalize','--','Phase Boundary Testing/Results/Deep_Clustering/run_20260908'],cwd=r,check=True)
verify_files=files+[t/'artifact_inventory.json']
requests=''.join(':'+p.relative_to(r).as_posix()+'\n' for p in verify_files).encode()
output=subprocess.check_output(['git','cat-file','--batch'],input=requests,cwd=r)
offset=0
for p in verify_files:
    end=output.index(b'\n',offset)
    header=output[offset:end].split()
    assert header[1]==b'blob',header
    size=int(header[2]);blob=output[end+1:end+1+size];offset=end+size+2
    assert hashlib.sha256(blob).digest()==hashlib.sha256(p.read_bytes()).digest(),p
print(f'All {len(files)+1} staged result files verified byte-for-byte, including inventory.')
