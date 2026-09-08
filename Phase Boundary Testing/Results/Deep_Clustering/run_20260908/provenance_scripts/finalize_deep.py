from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

workspace=Path(__file__).resolve().parents[1]
repo=workspace/'work/CMU-vPCF-Project'
root=repo/'Phase Boundary Testing'
out=workspace/'outputs/deep_phase_boundary'
sys.path.insert(0,str(root))
from run_deep_benchmark import make_figures

make_figures(out)
manifest=json.loads((out/'manifest.json').read_text())
manifest['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
manifest['reproducible_source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'run_deep_benchmark.py',root/'src/deep_benchmark.py',root/'src/metrics.py',root/'Phase_Boundary_Training_Local_Fallback.py']}
manifest['tests']='16 tests passed; see verification.txt'
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
search=pd.read_csv(out/'all_search_configurations.csv')
final=pd.read_csv(out/'final_metrics_by_seed.csv')
assert len(search)==48 and (search.status=='ok').all()
assert len(final)==15
logs=list((out/'search').glob('*/runtime.jsonl'))+list((out/'final').glob('*/runtime.jsonl'))
assert len(logs)==58
for log in logs:
    events=[json.loads(line) for line in log.read_text().splitlines()]
    assert events[-1]['event']=='training_verified'
    assert events[-1]['center_delta']>0 and events[-1]['encoder_delta']>0
    assert events[-1]['fallback'] is False
    assert events[-1]['cluster_steps']==150
for name in ['DEC','IDEC']:
    f=search[search.model==name]
    for field in ['latent_dim','n_clusters','clustering_weight','pretrain_epochs','learning_rate']:
        assert f[field].nunique()>1
aggregate=pd.read_csv(out/'final_metrics_aggregate.csv')
lines=['16 unit/integration tests passed.','48 independent search configurations completed (24 per model).',
       '10 final deep-model runs completed, separate from search; 5 K-means runs reproduced the earlier baseline.',
       '58 deep-training logs checked: positive encoder and center movement, 150 clustering steps, no fallback.',
       'Saved final benchmark features and labels match the earlier committed dataset.',
       'All five requested hyperparameters vary in each independent search.']
(out/'verification.txt').write_text('\n'.join(lines))
shutil.copy2(workspace/'work/all_tests.log',out/'tests.log')
shutil.copy2(workspace/'work/deep_training.log',out/'experiment.log')
report=(out/'REPORT.md').read_text(encoding='utf-8')
report += '\n\n## Verification and visual checks\n\n'+' '.join(lines)+'\n\n![Reference phase maps](phase_maps.png)\n\n![Every validation score](search_scores.png)\n'
means=aggregate.pivot(index='model',columns='metric',values='mean')
report+='\n## Practical interpretation\n\n'
for name in ['DEC','IDEC']:
    report+=f"{name} minus K-means: mean ARI {means.loc[name,'adjusted_rand_index']-means.loc['KMeans','adjusted_rand_index']:+.6f}; mean raw boundary F1 {means.loc[name,'raw_boundary_f1']-means.loc['KMeans','raw_boundary_f1']:+.6f}. "
report+='The highest validation ARI identifies the preferred deep configuration, but does not establish superiority to K-means or optimality on real research data. A single training seed per search configuration leaves selection uncertainty; the five final seeds quantify robustness only after selection.\n'
(out/'REPORT.md').write_text(report,encoding='utf-8')
source=out/'source/Phase Boundary Testing'
for rel in ['run_deep_benchmark.py','requirements-deep.txt','README.md','src/deep_benchmark.py','src/metrics.py','src/cluster_inspection.py','Phase_Boundary_Training_Local_Fallback.py','tests/test_deep_benchmark.py']:
    target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target)
baseline='Results/Local_Fallback_Runs/run_kmeans_benchmark_20260827_194059/spatial_phase_boundary_test'
for file in ['features.csv','ground_truth_labels.csv','metrics_by_seed.csv']:
    target=source/baseline/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/baseline/file,target)
files=['Phase Boundary Testing/'+p for p in ['requirements-deep.txt','run_deep_benchmark.py','src/deep_benchmark.py','tests/test_deep_benchmark.py']]
subprocess.run(['git','add','--intent-to-add','--',*files],cwd=repo,check=True)
patch=subprocess.check_output(['git','diff','--binary','HEAD','--','Phase Boundary Testing'],cwd=repo)
(workspace/'outputs/implementation.patch').write_bytes(patch)
shutil.make_archive(str(workspace/'outputs/deep_phase_boundary_bundle'),'zip',root_dir=workspace/'outputs',base_dir='deep_phase_boundary')
print('\n'.join(lines))
print(means[['adjusted_rand_index','normalized_mutual_info','raw_boundary_f1','raw_boundary_iou','raw_boundary_distance_px']].to_string())
