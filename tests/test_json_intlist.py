import numpy as np
import json

with open("/Users/liy/Documents/1.projects/LiyCg-github-blog/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

import numpy as np


def compute_delta(data, ref, norm_thresh=None):
    """
    compute the delta between a vector data and a ref vector

    norm_thresh allows to remove some outsiders. When markers doesn't exit, Nexus set the value to 0, therefore applying
    the norm_thresh will set this delta to 0

    :param data:
    :param ref:
    :param norm_thresh:
    :return:
    """
    deltas = []
    for d in data:
        delta = d - ref
        if norm_thresh is not None:
            delta[np.linalg.norm(delta, axis=1) > norm_thresh] = 0

        # check if delta is not filled by only zero -> != ref
        if np.any(delta):
            deltas.append(delta)

    return np.array(deltas)


lists = np.array(config['vrts_pos']).astype(int)
ref_actor_pose = np.array([[1,2,3],
                           [5,7,2],
                           [4,0,-2],
                           [-9,8,3],
                           [-1,2,-5],
                           [10,15,-11]])
ref = np.array([3,4,5])
print(compute_delta(ref_actor_pose, ref))
print(ref_actor_pose[[-4, -3]])
mean_pos = np.mean(ref_actor_pose[[-4, -3], :], axis=0)
print(mean_pos)
print(lists)




