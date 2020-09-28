import numpy as np
from scipy import optimize

from utils.load_data import load_training_frames
from utils.compute_delta import compute_delta
from utils.re_order_delta import re_order_delta
from src.compute_corr_coef import compute_corr_coef
from src.compute_corr_coef import compute_tilda_corr_coef
from src.compute_trust_values import compute_trust_values
from utils.get_key_expressions import get_key_expressions
from src.get_soft_mask import get_soft_mask
from src.EAlign import EAlign
from src.RBF_warp import get_initial_actor_blendshapes
from plotting import plot_similarities


"""
run: python -m blendshape_transfer
"""

# define parameters
max_num_seq = 3  # set to None if we want to use all the sequences
do_plot = True

# load data
sk = np.load('data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
ref_sk = sk[-1]  # neutral pose is the last one
delta_sk = compute_delta(sk[:-1], ref_sk)
delta_sk = delta_sk[:25]  # todo remove here to train on all bs!
af, delta_af = load_training_frames('D:/MoCap_Data/David/NewSession_labeled/', num_markers=45, max_num_seq=max_num_seq)  # actor animation  # todo downsamples freq?
af = np.delete(af, (38, 39, 40, 44), 1)  # remove HEAD markers
delta_af = np.delete(delta_af, (38, 39, 40, 44), 1)  # remove HEAD markers

print("Finished loading data")
print("shape af, sk:", np.shape(af), np.shape(sk))
print("shape delta_sk", np.shape(delta_sk))
print("shape delta af:", np.shape(delta_af))
print()
K, M, n_dim = np.shape(delta_sk)
F = np.shape(delta_af)[0]

# 1) Facial Motion Similarity
# reorder delta blendshapes
sorted_delta_sk = re_order_delta(np.reshape(delta_sk, (np.shape(delta_sk)[0], -1)))
print("shape sorted_delta_sk", np.shape(sorted_delta_sk))

# measure similarity between character blendshapes and actor's capture performance
ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                        np.reshape(sorted_delta_sk, (np.shape(delta_sk)[0], -1)))
if do_plot:
    plot_similarities(ckf, "Fig. 7: Motion space similarity")

# contrast enhancement
tk = compute_trust_values(sorted_delta_sk, do_plot=do_plot)
tilda_ckf = compute_tilda_corr_coef(ckf, tk)
print("shape ckf", np.shape(ckf))
print("shape tk", np.shape(tk))
print("shape tilda_ckf", np.shape(tilda_ckf))
print()

# 2) Key Expression Extraction
key_expressions = get_key_expressions(tilda_ckf, ksize=3, theta=2, do_plot=do_plot)
print("shape key_expressions", np.shape(key_expressions))  # todo where is it used?!?!?
print()

# 3) Manifold Alignment
# built soft max vector
uk = get_soft_mask(delta_sk)
print("shape uk", np.shape(uk))

# 4) Geometric Constraint
# build initial guess blendshape using RBF wrap (in delta space)
delta_gk = get_initial_actor_blendshapes(ref_sk, af[0], delta_sk)

# 5) Cross-Expression Constraint (Cross-Expression Graph: CEG)
delta_sk = np.reshape(delta_sk, (K, M*n_dim))

# 6) declare E_Align
e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, delta_sk)

# 7) build personalized actor-specific blendshapes
