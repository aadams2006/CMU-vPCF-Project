import h5py
from pathlib import Path

h5_path = Path(r"Experimentally-obtained vPCF Testing/1730_poled_TaN_cropped/data/1730_poled_TaN_cropped_100.h5")

with h5py.File(h5_path, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    print("\nExploring structure:")
    
    def explore_group(name, obj):
        indent = "  " * (name.count('/') - 1)
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: {obj.shape} {obj.dtype}")
        else:
            print(f"{indent}{name}/")
    
    f.visititems(explore_group)
