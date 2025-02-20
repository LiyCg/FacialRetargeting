import numpy as np
import time as time
import os
import json
import matplotlib.pyplot as plt

from utils.load_data import load_training_frames
"""
def compute_delta(data, ref, norm_thresh=None):

    compute the delta between a vector data and a ref vector
    norm_thresh allows to remove some outsiders. When markers doesn't exit, Nexus set the value to 0, therefore applying
    the norm_thresh will set this delta to 0
"""
from utils.compute_delta import compute_delta
from utils.remove_neutral_blendshape import remove_neutral_blendshape
from utils.normalize_positions import normalize_positions
from utils.align_to_head_markers import align_to_head_markers
from utils.modify_axis import modify_axis
from utils.re_order_delta import re_order_delta
from utils.get_key_expressions import get_key_expressions
from src.compute_corr_coef import compute_corr_coef
from src.compute_corr_coef import compute_tilda_corr_coef
from src.compute_trust_values import compute_trust_values
from src.get_soft_mask import get_soft_mask
from src.EAlign import EAlign
from src.RBF_warp import get_initial_actor_blendshapes
from utils.plotting import plot_similarities


"""
run: python -m blendshape_transfer
"""
# precision=4 : 출력 소숫점 4자리까지
# linewidth=200 : line 별로 200 char 씩 출력
# suppress=True : 참일때, always print floating point numbers using fixed point notation
np.set_printoptions(precision=4, linewidth=200, suppress=True)

# load and define parameters
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# alpha: 1.0 / beta: 1.0으로 설정 
alpha = int(config['alpha'])
beta = int(config['beta'])
print("[PARAMS] alpha:", alpha)
print("[PARAMS] beta:", beta)
print("[PARAMS] sk loaded:", config['vertices_pos_name'])

max_num_seq = None  # set to None if we want to use all the sequences
do_plot = True
save = True
load_pre_processed = True

# load data
mesh_list = np.load(os.path.join(config['python_data_path'], config['maya_bs_mesh_list']+'.npy')).astype(str)
    
    # data folder 안에 존재, marker label 별로 vertex id(from vk)가 value값으로 dictionary로 정의되어있다.
    # sparse representation of all blendshapes(vk)
    # blendshape들은 동일한 mesh structure를 가지므로, sk는 1:1 대응하는 correspondence data만 갖고있다. 
sk = np.load(os.path.join(config['python_data_path'], config['vertices_pos_name']+'npy'))  

# get Neutral ref index and new cleaned mesh list
"""
def remove_neutral_blendshape(mesh_list, neutral_pose_name):

(설명)   mesh_list에서 neutral_pose 메쉬를 찾아서 지운 새로운 mesh_list와 
        새로운 (int) 인덱스리스트 그리고 지운 neutral mesh의 인덱스 반환 
"""
cleaned_mesh_list, bs_index, ref_index = remove_neutral_blendshape(mesh_list, config['neutral_pose'])

# get neutral (reference) blendshape and normalize it
"""
def normalize_positions(pos, min_pos=None, max_pos=None, return_min=False, return_max=False):

(설명)    주어진 pos 값(x,y,z) 중 min/max 값 찾아서 vertex별 해당값으로 복사하고, 같은 shape의 vector/matrix로 normalize.
         return_min과 return_max가 True일 경우(모두), normalize할 때 쓴 min/max matrix/vector도 return한다. 
"""
# np.copy(a): return copy of given array
ref_sk, min_sk, max_sk = normalize_positions(np.copy(sk[ref_index]), return_min=True, return_max=True)

# normalize sk(using the same 'min_pos' and 'max_pos' matrix used to normalize neutral sk
sk = normalize_positions(sk, min_pos=min_sk, max_pos=max_sk)

# sk를 3차원으로 sk를 mesh로 plot
if do_plot:
    # figure를 만들고 편집할 수 있게 만들어주는 함수
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(ref_sk[:, 0], ref_sk[:, 1], ref_sk[:, 2])
    ax.set_title("Ref sk")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# compute delta sparse blendshape
    # this delta sparse blendshape is a displacement field for activating a particular expressions
    # new facial poses are computed by displacing weighted sum of these delta blendshapes, with weights
# [:] : first elem to last elem
delta_sk = compute_delta(sk[bs_index, :, :], ref_sk)

# test if delta_sk has no none-unique entry
# np.unique(): 중복 제거하고 unique한 값으로만 표현된 np array 생성(column별) 
test_unique = np.unique(delta_sk, axis=1)
if np.shape(test_unique)[0] != np.shape(delta_sk)[0]:
    raise ValueError("delta_sk contains non unique entry! Check your index dictionary to build the sparse blendshape "
                     "(maya_scripts.03_extract_blendshapes_pos_and_obj)")

# ----------------------------------------------------------------------------------------------------------------------
# get Actor Animation
# ----------------------------------------------------------------------------------------------------------------------

# load ref pose of an actor
# david이 actor이고, louise가 캐릭터 이름인듯하다. 참고하
ref_actor_pose = np.load(os.path.join(config['python_data_path'], config['neutral_pose_positions']+'.npy'))
# align sequence with the head markers
# range(A,B) : A,  B-1까지 정수를 list로 return 
# marker 들 중 끝에서 -4, -3, -2  index를 가져옴 -> 이게 head marker인듯(Head1, Head2, Head3)
head_markers = range(np.shape(ref_actor_pose)[0] - 4, np.shape(ref_actor_pose)[0] - 1)  # use only 3 markers
"""
def align_to_head_markers(positions, ref_idx, roll_ref=25):
    
    Align the position to 3 markers (ref_idx). The function correct the rotation and position, as to fix the center
    of the 3 markers to zero and having 0 angles.
    roll_ref allows to set a desired angle
    :param pos: positions
    :param ref_idx: (3,)
    :param roll_ref: int definying what angle we want the roll to be
    :return:

"""
# ref_actor_pose를 template head_markers의 위치로 align시킨다.
ref_actor_pose = align_to_head_markers(ref_actor_pose, ref_idx=head_markers)


if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
    ax.set_title("ref pose A0")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

ref_actor_pose = ref_actor_pose[:-4, :]  # remove HEAD markers
# modify axis from xzy to xyz to match the scatter blendshape axis orders
ref_actor_pose = modify_axis(ref_actor_pose, order='xzy2xyz', inverse_z=True)
# normalize reference (neutral) actor positions
ref_actor_pose, min_af, max_af = normalize_positions(ref_actor_pose, return_min=True, return_max=True)

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
    ax.set_title("ref pose A0 normalized")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # it's set to true above
if load_pre_processed:
    delta_af = np.load("data/training_delta_af_v2.npy")
    tilda_ckf = np.load("data/training_tilda_ckf_v2.npy")
else:
    # load sequence
    # max_num_seq : set to None if we want to use all the sequences
    # selected training frames that are semantically equivalent with all the sk..
    # is this manually selected...?
    af = load_training_frames(config['mocap_folder'],
                              num_markers=int(config['num_markers']),
                              template_labels=config['template_labels'],
                              max_num_seq=max_num_seq,
                              down_sample_factor=5)
    af = align_to_head_markers(af, ref_idx=head_markers)
    af = af[:, :-4, :]  # remove HEAD markers
    # modify axis from xyz to xzy to match the scatter blendshape axis orders
    af = modify_axis(af, order='xzy2xyz', inverse_z=True)
    af = normalize_positions(af, min_pos=min_af, max_pos=max_af)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
        ax.scatter(af[0, :, 0], af[0, :, 1], af[0, :, 2], c='RED')
        # ax.scatter(af[5, :, 0], af[5, :, 1], af[5, :, 2], c='RED')
        # ax.scatter(af[10, :, 0], af[10, :, 1], af[10, :, 2], c='RED')
        # ax.scatter(af[2575, :, 0], af[2575, :, 1], af[2575, :, 2], c='YELLOW')
        ax.set_title("ref_pose A0 vs. af[0]")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    delta_af = compute_delta(af, ref_actor_pose, norm_thresh=2) 

    print("[data] shape af:", np.shape(af))

print("[data] Finished loading data")
print("[data] Neutral Blendshape index:", ref_index)
print("[data] shape ref_actor_pose", np.shape(ref_actor_pose))
print("[data] shape delta af:", np.shape(delta_af))
print("[data] shape sk", np.shape(sk))
print("[data] shape delta_sk", np.shape(delta_sk))
print("[data] cleaned_mesh_list:", len(cleaned_mesh_list))

# get dimensions
K, M, n_dim = np.shape(delta_sk)
F = np.shape(delta_af)[0]
print("[data] num_blendshapes:", K)
print("[data] num_markers:", M)
# b/c we optimize all personalized delta-blendshapes pk, all xyz vertex position values are 'features'
print("[data] num_features (M*3):", M*n_dim)
print("[data] num_frames", F)
print()

####################################### 4/11 done

# 1) Facial Motion Similarity
"""
similarity of the ranges of motion is best visualized by considering the displacement of delta_af(af-a0)
over the entire sequence and the delta-bshp(displacement) of the corresponding vertex on the facial rig (delta_sk=sk-s0)

-> 왜 Similarity를 정량화하는가? 
    similarity를 계산해서 최대한 비슷한 bshp만을 뽑아내어 parallel parameterization해야함. 
-> 어떻게 정량화하는가?
    af와 sk 간에 Pearson Correlation을 계산한다. 다만, mean은 neutral pose로 대체하고, 
    그렇게되면 각각 delta_af 와 delta_sk로 비교적 손쉽게 표현가능하게 된다. 

"""
# reorder delta blendshapes
    """
    re-order delta matrix data according to its norm
    the re-ordering goes from the bigger to smallest norm difference(default: 2-norm)
    :param data:
    :return:
    """
sorted_delta_sk, sorted_index = re_order_delta(delta_sk)
# neutral mesh를 빼고 정리한 af mesh들 list를 norm 별로 sorting한 sk와 동일 순서로 재배열 
# list 예를 들면, (0,2,1) 를 np.array에 명시하면, 그 index대로 순서가 바뀜..;; 개쩌는 기능이다. 
sorted_mesh_list = np.array(cleaned_mesh_list)[sorted_index]
print("[Pre-processing] shape sorted_delta_sk", np.shape(sorted_delta_sk)) # K X M X 3
print("[Pre-processing] len sorted_mesh_list", len(sorted_mesh_list)) # K-1 (neutral pose빼고)

# load_pre_processed == False 일때, 본격 similarity 계산 시작
if not load_pre_processed:
    # measure similarity between character blendshapes(sk) and actor's capture performance(af)
    """
    def compute_corr_coef(da, ds):
   
        Compute Pearson Correlation Coefficient between a and s
        This is formula 5 of the paper
        f:= number of frames
        k:= number of blendshapes
        n:= num_features (n_markers*3)
        :param da: delta-actor training sequence (f, n)
        :param ds: delta-character sparse blendshapes (k, n)
        :return: correlation coefficients in delta representation
    """
    # why reshape..? 어차피 똑같은 shape으로 변형하는데...?
    ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                            np.reshape(sorted_delta_sk, (np.shape(sorted_delta_sk)[0], -1)))
    
    # heatmap plotting using 'Seaborn' package
    if do_plot:
        plot_similarities(ckf, "Fig. 7: Motion space similarity")

    # contrast enhancement - (Similarity value 한층 강화하는 방법) 
    """
    How its done : 
    
    trust-value를 sk끼리의 correlation으로 구해서 t-value를 구한다.
    구한 t-value는 위에서 구한 similarity를 더욱 강화하기 위해 사용된다, important bshp(sk)들만 골라내는것이다! 
    t값이 작을수록 해당 sk 둘은 correlation이 큰 것이고, 둘 중 하나는 필요가 없다 
    고로, 
    t값이 작 애들에대해서는 original ckf쪽으 더 값이 치우치게, 
    t값이 큰 애들에 대해서는 amplified ckf쪽으로 값이 치우치게 한다. 
  
    """
    """
    def compute_trust_values(dsk, do_plot=False):
    
    Compute trust values following formula 6
    k:= number of blendshapes
    n:= num_features (num_markers*3)
    :param dsk: delta_sk vector (k, n)
    :param do_plot: decide if we want to plot the between-correlation matrix
    :return: trust values vector (k,)
    """    
    tk = compute_trust_values(np.reshape(sorted_delta_sk, (np.shape(sorted_delta_sk)[0], -1)), do_plot=do_plot)
    """
    def compute_tilda_corr_coef(ckf, tk, r=15):
   
    compute similarity tilda_ckf from equation 8 in the paper
    original ckf와 modified(amplified)된 b(ckf) 사이에서 t-val로 선형보간하는 것이다
    k:= number of blendshapes
    f:= number of frames
    :param ckf: correlation matrix from compute_corr_coef (k, f)
    :param tk: trust values (k,)
    :param r: steepness
    :return:
    """
    tilda_ckf = compute_tilda_corr_coef(ckf, tk)
    print("[Pre-processing] shape ckf", np.shape(ckf))
    print("[Pre-processing] shape tk", np.shape(tk))
    print("[Pre-processing] shape tilda_ckf", np.shape(tilda_ckf))
    print()

    # 2) Key Expression Extraction
    """
    def get_key_expressions(sequence, ksize=3, theta=1, do_plot=False):
    
    Extract key expressions as in 4.2 Key Expression Extraction
    
    1) apply low pass filtering over each row
    2) sum filtered correlation coefs over column(per frame)
    3) extract local peaks(frame index) over sumed up vector
    
    k:= number of blendshapes
    f:= number of frames
    :param sequence: input data (k, f)
    :param ksize: int parameter to define the size of the 1D Gaussian kernel size
    :param theta: float parameter to define the Gaussian filter
    :param do_plot: option to plot the cumulative correlation as in Fig. 8
    :return: key expressions within sequence
    """
    key_expressions_idx = get_key_expressions(tilda_ckf, ksize=3, theta=2, do_plot=do_plot)
    F = len(key_expressions_idx)
    # key_expressions_idx로 해당 af만 추출  
    delta_af = delta_af[key_expressions_idx, :, :]
    # ckf(correlation coef)도 해당 af들과의 corr 값들만 가져와서 행렬 재추출
    tilda_ckf = tilda_ckf[:, key_expressions_idx] 
    print("[Key Expr. Extract.] Keep", F, "frames")
    print("[Key Expr. Extract.] shape key_expressions", np.shape(key_expressions_idx))
    print("[Key Expr. Extract.] shape delta_af", np.shape(delta_af))
    print("[Key Expr. Extract.] shape tilda_ckf", np.shape(tilda_ckf))
    print()
    # saving as a file
    np.save("data/training_delta_af", delta_af)
    np.save("data/training_tilda_ckf", tilda_ckf)

####################################### 4/12 done ( utils/get_key_expressions.py 참고)
    
# 3) Manifold Alignment
"""

goal is to fit the personalized bshp delta pk to the delta af
    : meaning closed-eye bshp(sk) should be fitted to an closed eye expression in actor's sequence,
    not being influenced by the negatively correlated eye-opening movements. 
However, without proper consideration of local support of each bshps,, the alignment is distributed
SO we encode local support with a soft mask vector uk for bshp k(sk). 

uk : entries corresponding to the x/y/z coordi- of marker 'm' of bshp 'k'(sk), are computed as 
norm of (delta of marker position of sk) divided by largest marker displacement within bshp delta sk.

"""
# built soft max vector 
# this vector's shape is (k, mX3) and each 3 position value is 
# all same three norm, of each markers divided by the max norm among corresponding sk
    # I guess not applying just norm but by dividing with the max norm of each sk leads to 'soft' masking
# 결국, local한 bshp들의 '조합'과 align되는 actor's animation 같은 경우, 두 local bshp들로 alignment가 분산되게 된다는 것!
# 그렇기 때문에 max_norm을 구하여 각 norm들의 비율을 각 xyz에 곱해준다는 것은
# 가장 큰 displacement만 남기고 다른 displacement는 작게 만들어버린다는 것이다!
# max_norm에 해당하는 Marker만 1/1/1 이 그 marker의 xyz에 곱해질것이고 나머지는 전부 1보다 작은 0보다 큰 만큼의 fraction이 곱해지니깐!
uk = get_soft_mask(sorted_delta_sk)
print("[SoftMax] shape uk", np.shape(uk)) # (k, m*3)
print()

####################################### 4/13 done ( src/get_soft_mask.py 참고 )

# 4) Geometric Constraint
# build initial guess blendshape using RBF wrap (in delta space)\
# purpose: 
#       preserving local features of an expression
#       ex) o-shape of the lips for the kiss expression, is the intention of the geometric constraint
# process: 
# 		Based on sparse correspondence data btw s0(of the charater rig), and a0(of an actor),
# 		1) create initial guess gk, for each personalized bshp pk, using RBF wrap(in delta space)
#		2) first compute RBF thin-plate spline that transforms netural s0 to a0, 
#			by placing RBF center at every marker of s0 and solving for RBF weights
# 		3) resulting RBF func is used to convert delta sk to initial guess delta gk of actor bshp
#
#	* geometric prior 의 필요성 : personalized bshp인 pk를 만드는데 있어, local shape properties를 보존하기 위해
#							-> geo- prior란 미리 준비된 high-resolution generic face mesh를 말하는듯하다.  	

# 내가 이해한 내용 -
# pk에 대응하는 gk 만들기 위해, 각 (extract한) frame별로 RBF function을 적용하는 것이다.
# s0에서 a0로 변환하는 RBF thin plate spline을 구해서 각 marker 별로 RBF weight를 구하고 그 weight들로 구성된 함수 RBF function을 완성.
# 그 함수 delta sk 적용하면에 그것이 delta gk인것 를

"""
def get_initial_actor_blendshapes(s0, a0, delta_sk):

Compute the initial guess(gk) actor blendshapes in delta space as explained in 4.4 Geometric Constraint of the paper
k:= num_of_blendshapes
m:= num_of_markers
:param s0: neutral character expression
:param a0: neutral actor expression
:param delta_sk: character blendshapes in delta space
:return: initial guess of actor blendshapes 
"""
delta_gk = get_initial_actor_blendshapes(ref_sk, ref_actor_pose, sorted_delta_sk)
print("[RBF Wrap] shape delta_gk", np.shape(delta_gk)) 
print()

# plotting 'delta sk vs. initial actor blendshape gk'
if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    bs_idx = 0
    sk0 = ref_sk + sorted_delta_sk[bs_idx]
    ax.plot_trisurf(sk0[:, 0], sk0[:, 1], sk0[:, 2], alpha=0.6)
    gk0 = ref_sk + delta_gk[bs_idx]
    ax.plot_trisurf(gk0[:, 0], gk0[:, 1], gk0[:, 2], alpha=0.6)
    ax.set_title("delta sk[{}] vs. initial actor blendshape gk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

####################################### 4/14 done ( src/RBF_warp.py 참고 )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 5) build personalized actor-specific blendshapes (delta_p)
# reshape to match required dimensions( not blocks, convert to matrices )
delta_af = np.reshape(delta_af, (F, M*n_dim)) # K, M, n_dim = np.shape(delta_sk) / F = len(key_expressions_idx)
sorted_delta_sk = np.reshape(sorted_delta_sk, (K, M*n_dim))
# print control of all shapes
print("[dp] shape tilda_ckf:", np.shape(tilda_ckf)) 
print("[dp] shape uk:", np.shape(uk)) # (k, m*3)
print("[dp] shape delta_af:", np.shape(delta_af))
print("[dp] shape delta_gk:", np.shape(delta_gk))
print("[dp] shape delta_sk", np.shape(sorted_delta_sk))

# declare E_Align
"""
def __init__(self, tilda_ckf, uk, delta_af, delta_gk, ref_sk, delta_sk, alpha=0.01, beta=0.1):

Construct a class to compute E_Align as in formula 4 using a function to pass directly the personalized blendshapes
in delta space delta_p (dp)
k:= num_of_blendshapes
f:= num_frames
m:= num_markers
n:= num_features (n_m * 3)
"""
# alpha: 1.0 / beta: 1.0으로 설정 
e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, ref_sk, sorted_delta_sk, alpha=alpha, beta=beta)
# compute personalized actor-specific blendshapes
start = time.time()
"""
def compute_actor_specific_blendshapes(self, vectorized=True):
 
Solve EAlign to compute the personalized actor-specific blendshapes in delta space (delta_p)
The function solve the system Ax + b for each xyz coordinates and merge the results
:return: delta_p (n_k*n_n, ) 
"""
###########################################################
# Here's the pk what i was wondering desperately about!!
delta_p = e_align.compute_actor_specific_blendshapes(vectorized=False) #personalized bshp pk returned in matrix form(b/c vectorized=False)
###########################################################
print("[dp] Solved in:", time.time() - start)
print("[dp] shape delta_p", np.shape(delta_p))
print() 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

####################################### 4/15 done ( src/EAlign.py 참고 )

# 6) save delta_p and sorted_mesh_list
if save:
    saving_name = config['dp_name']+'_alpha_'+config['alpha']+'_beta_'+config['beta']
    np.save(os.path.join(config['python_data_path'], saving_name), delta_p)
    np.save(os.path.join(config['python_data_path'], config['sorted_maya_bs_mesh_list']), sorted_mesh_list)
    print("[save] saved delta_pk (actor specific bshps), shape:", np.shape(delta_p))
    print("[save] saved sorted_mesh_list, shape:", np.shape(delta_p))

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    gk0 = ref_sk + delta_gk[bs_idx]
    ax.plot_trisurf(gk0[:, 0], gk0[:, 1], gk0[:, 2])
    delta_p = np.reshape(delta_p, (K, M, n_dim))
    pk0 = ref_sk + delta_p[bs_idx]
    ax.plot_trisurf(pk0[:, 0], pk0[:, 1], pk0[:, 2], alpha=0.6)
    ax.set_title("initial dgk[{}] vs. optimized dpk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(sk0[:, 0], sk0[:, 1], sk0[:, 2], alpha=0.6)
    ax.plot_trisurf(pk0[:, 0], pk0[:, 1], pk0[:, 2], alpha=1.0)
    ax.set_title("sk[{}] vs. optimized dpk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

if do_plot:
    plt.show()
