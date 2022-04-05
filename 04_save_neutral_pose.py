import numpy as np
import os
"""
    C3DReader, imported inside load_data.py,
    is a package that provides a single Python module 
    for reading and writing binary motion-capture files in the C3D file format.
"""
from utils.load_data import load_c3d_file
import json

# load and define parameters 
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load sequence
"""
    function 'load_c3d_file()'
    Load c3d file of interest and transform it into a list and labels
    The function can take in argument a template labels list that is used to sort the positions accordingly
    :param file: .c3d file of interest
    :param template_labels: sort the position according to the template given
    :param get_labels:
    :param verbose: print option
    :return: 
        1) if get_labels:
            numpy array of all 3D position (x, y, z) for each frames/markers and labels of markers (n_frames, n_markers, 3)
        2) else:
             numpy array of all 3D position (x, y, z) for each frames/markers
"""
# data: frame lists
# labels: marker labels
data, labels = load_c3d_file(os.path.join(config['mocap_folder'], config['neutral_sequence']),
                             template_labels=config['template_labels'],
                             get_labels=True,
                             verbose=True)

# checkups
print("labels", len(labels))
print(labels)
print("shape of data[neutral_frame]", np.shape(data[int(config['neutral_frame'])])) #(1, n_markers, 3)
print(data[int(config['neutral_frame'])]) #print out the list of above (1, n_markers, 3) np.array

# save neutral frame's marker positions for each marker
np.save(os.path.join(config['python_data_path'], config['neutral_pose_positions']), data[int(config['neutral_frame'])])
