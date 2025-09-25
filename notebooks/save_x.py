import json; import os; import numpy as np; data_dir = '../data/saved_vpcfs'; vpcfs = [];
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            for key in data:
                vpcfs.extend(data[key]['$array'])
x = np.array(vpcfs)
np.save('x.npy', x)