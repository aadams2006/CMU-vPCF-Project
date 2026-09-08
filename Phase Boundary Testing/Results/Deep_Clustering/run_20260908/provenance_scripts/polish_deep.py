from pathlib import Path
import json
import hashlib
import shutil
import subprocess

w=Path(__file__).resolve().parents[1]
r=w/'work/CMU-vPCF-Project';s=r/'Phase Boundary Testing';o=w/'outputs/deep_phase_boundary'
p=s/'run_deep_benchmark.py'
text=p.read_text(encoding='utf-8')
text=text.replace("'Separate searches: 24 configurations per model by default, independently sampled with seeds 101 (DEC) and 202 (IDEC), plus the default configuration.", "f'Separate searches: {manifest[\"trials_per_model\"]} configurations per model, including one default configuration; the others are independently sampled with seeds 101 (DEC) and 202 (IDEC).")
p.write_text(text,encoding='utf-8')
compile(text,str(p),'exec')
report=(o/'REPORT.md').read_text(encoding='utf-8')
report=report.replace('24 configurations per model by default, independently sampled with seeds 101 (DEC) and 202 (IDEC), plus the default configuration.', '24 configurations per model: 23 random draws and one default, independently sampled with seeds 101 (DEC) and 202 (IDEC).')
report+='\nIDEC retained three occupied clusters in all five final runs, with all cluster fractions between 30.6% and 38.2%. DEC seed 46 had sizes [1708, 988, 376] (55.6%, 32.2%, 12.2%), ARI 0.51056, and boundary F1 0.32696. It passes the narrow empty/tiny-cluster check but is a poor and imbalanced phase recovery; occupied count alone is insufficient. IDEC is the preferred deep model because its validation ARI is highest and its final seed variability is substantially lower. K-means remains the best measured method on this benchmark.\n'
(o/'REPORT.md').write_text(report,encoding='utf-8')
shutil.copy2(p,o/'source/Phase Boundary Testing/run_deep_benchmark.py')
m=json.loads((o/'manifest.json').read_text())
m['reproducible_source_sha256']['run_deep_benchmark.py']=hashlib.sha256(p.read_bytes()).hexdigest()
(o/'manifest.json').write_text(json.dumps(m,indent=2))
(w/'outputs/implementation.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD','--','Phase Boundary Testing'],cwd=r))
shutil.make_archive(str(w/'outputs/deep_phase_boundary_bundle'),'zip',root_dir=w/'outputs',base_dir='deep_phase_boundary')
print('Report, runnable source archive, and implementation patch finalized.')
