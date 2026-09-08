"""Preserve all experiment deliverables in the phase-boundary repository folder."""
from pathlib import Path
import shutil
import hashlib
import json

workspace=Path('\\\\?\\'+str(Path(__file__).resolve().parents[1]))
phase=workspace/'work/CMU-vPCF-Project/Phase Boundary Testing'
target=phase/'Results/Deep_Clustering/run_20260908'
target.mkdir(parents=True,exist_ok=True)
original=workspace/'outputs/deep_phase_boundary'
shutil.copytree(original,target,dirs_exist_ok=True)
for filename in ['deep_phase_boundary_bundle.zip','implementation.patch']:
    shutil.copy2(workspace/'outputs'/filename,target/filename)
provenance=target/'provenance_scripts'
provenance.mkdir(exist_ok=True)
for filename in ['finalize_deep.py','polish_deep.py','publish_results.py']:
    shutil.copy2(workspace/'work'/filename,provenance/filename)
shutil.copy2(workspace/'work/deep_tests.log',target/'initial_test_attempt.log')
(provenance/'README.md').write_text(
    '# Historical packaging scripts\n\nThese scripts preserve the original workstation packaging process. '
    'They expect the original workspace layout and are not the experiment entry point. '
    'To reproduce training, use `Phase Boundary Testing/run_deep_benchmark.py`. '
    '`initial_test_attempt.log` records an initial Windows temporary-file cleanup failure, '
    'fixed before the final 16-test passing run in `tests.log`.\n',encoding='utf-8')
for p in original.rglob('*'):
    if p.is_file():
        copied=target/p.relative_to(original)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(copied.read_bytes()).digest(),p
files=[p for p in target.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']
inventory=[dict(path=p.relative_to(target).as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
(target/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
readme=phase/'README.md'
text=readme.read_text(encoding='utf-8')
text+='\n## Completed DEC / IDEC experiment — 2026-09-08\n\n'
text+='[Full findings and every hyperparameter configuration](Results/Deep_Clustering/run_20260908/REPORT.md) '
text+='are saved with [all run artifacts](Results/Deep_Clustering/run_20260908/). '
text+='The archive includes 48 search runs, 10 final deep-model runs, five K-means baseline runs, '
text+='checkpoints, fitted centers, learned embeddings, synthetic labels, scores, uncertainty, plots, '
text+='runtime logs, source snapshots, and an SHA-256 inventory. '
text+='All 16 final tests passed. IDEC is the preferred deep model; K-means remains strongest '
text+='on this synthetic benchmark. No research H5/DM3 data were used.\n'
readme.write_text(text,encoding='utf-8')
print(f'Preserved {len(files)} files plus inventory; {sum(p.stat().st_size for p in files):,} bytes. All original deliverables match SHA-256.')
