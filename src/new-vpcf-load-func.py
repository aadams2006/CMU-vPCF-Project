from tqdm.notebook import tqdm
from pathlib import Path
import h5py

##FUNCTION

def read_vpcf_frame(filename, frame_index):
    """
    Read one frame (atomic_position, origin, peaks, image) from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        grp = f["experiments"]

        atomic_position = grp["atomic_positions"][frame_index]
        vpcf_origin = grp["vpcf_origin"][frame_index]

        flat_peaks = grp["vpcf_peaks"][frame_index]
        peaks_shape = grp["peaks_shapes"][frame_index]
        vpcf_peaks = flat_peaks.reshape(peaks_shape)

        vpcf_image = grp["vpcf_images"][frame_index]

    return atomic_position, vpcf_origin, vpcf_peaks, vpcf_image

##Implementation

file= "../data/new-vpcf-data.h5"
atomic_positions2 = []
vpcf_origin2 = []
vpcf_peaks2 = []
vpcf_image2 = []
for i in tqdm(range(7300)):
    atomic_position, vpcf_origin, vpcf_peaks, vpcf_image = read_vpcf_frame(Path(file), i)
    atomic_positions2.append(atomic_position)
    vpcf_origin2.append(vpcf_origin)
    vpcf_peaks2.append(vpcf_peaks)
    vpcf_image2.append(vpcf_image)