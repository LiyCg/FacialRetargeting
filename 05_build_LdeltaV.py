
"""
PyMesh is a rapid prototyping platform focused on geometry processing. 
It provides a set of common mesh processing functionalities 
and interfaces with a number of state-of-the-art open source packages 
to combine their power seamlessly under a single developing environment.
"""

import pymesh
import numpy as np
import os
import time
import json

# need it to run with python2 since pymesh was installed in my python2...
import sys
"""
sys.path : 
 - 모듈 import시에 찾아야할 경로들을 저장해둔 list
 - 'import utils/'라는 코드를 만나면 list 원소들 안에 찾으려는 해당 모듈이 존재하는지 차례 차례 점검한다.
 - sys.path의 첫번째 값([0])은 파이썬 스크립트가 '위치'한 directory(즉 어디에서 스크립트를 실행했는지 중요치 않음)
"""
# utils folder로 path를 변경한 것
# sys.path.insert(0,'utils/') : 스크립트가 위치한 dir에 utils/에 추가  
sys.path.insert(0, 'utils/')
# sys.path.append('utils/') : sys.path 마지막에 utils/ 를 추가
sys.path.append('utils/')

# utils folder 내부에서 'normalize_positions' file 접근
from normalize_positions import normalize_positions

"""

    - Delta blendshape representation

"""

# ref_mesh : neutral pose we extracted from c3d mocap data file (neutral frame was set manually)
def compute_deltaV(mesh, ref_mesh, faces):
    dv = mesh - ref_mesh
    
    # - for surface mesh:
    #   mesh = pymesh.form_mesh(vertices, faces)
    # - for volume mesh:
    #   mesh = pymesh.form_mesh(vertices, faces, voxels)
    # * vertices, faces and voxels are of type numpy.ndarray. One vertex/face/voxel per row.
    return pymesh.form_mesh(dv, faces)


# path: to load blendshape mesh/ref_mesh
def build_L_deltaV(mesh_list, path, ref_mesh_name):
    # pymesh.load_mesh() : 
    # PyMesh supports parsing the following formats: 
    # .obj, .ply, .off, .stl, .mesh, .node, .poly and .msh.
    ref_mesh = pymesh.load_mesh(os.path.join(path, ref_mesh_name + ".obj"))
    faces = ref_mesh.faces # output: [num_faces X 3], where all values are 'int', which is the index of vertices
    
    # ref_mesh_vertices, min_mesh, max_mesh = normalize_positions(np.copy(ref_mesh.vertices), return_min=True, return_max=True)
    ref_mesh_vertices = ref_mesh.vertices
    n_vertices = len(ref_mesh_vertices)
    print("n_vertices:", n_vertices)
    
    LdVs = []
    for mesh_name in mesh_list:
        # compute dV
        if mesh_name != ref_mesh_name:
            
            # load blendshape
            mesh = pymesh.load_mesh(os.path.join(path, mesh_name+".obj"))
            # mesh_vertices = normalize_positions(np.copy(mesh.vertices), min_pos=min_mesh, max_pos=max_mesh)
            mesh_vertices = mesh.vertices # output: [num_vertices X 3]
            dv_mesh = compute_deltaV(mesh_vertices, ref_mesh_vertices, faces) # output: surface pymesh
               
            ##############################################
            ## compute Laplacians(=geometry processing) ##
            ##############################################
            
            """
            class pymesh.Assembler(mesh, material=None) : finite element matrix assembler
            In digital geometry processing, one often have to assemble matrices 
            that corresponding to *'discrete differential operators'. 
            PyMesh provides a simple interface to assemble commonly used matrices.
            
            * discrete differential operators
            Geometry processing of surface meshes relies heavily on the discretization
            of differential operators such as gradient, Laplacian, and covariant derivative
            
            """
            assembler = pymesh.Assembler(mesh)
            """
            This example assembles the 'Laplacian-Beltrami' matrix used by many graphics applications.
            
            Other types of finite element matrices include:
            stiffness
            mass
            lumped_mass
            laplacian
            displacement_strain
            elasticity_tensor
            engineer_strain_stress
            rigid_motion
            gradient
            """
            L = assembler.assemble("laplacian").todense()

            # compute LdV 
            # np.dot() : 일반적인 행렬곱을 생각하면됨. vector끼리 곱도 가능. 
            LdVs.append(np.dot(L, dv_mesh.vertices)) # output: blendshape 별 laplacian으로 geometry processing된 mesh들
        else:
            print("[Warning] Ref blendshape found in the sorted mesh list name!")

    return np.array(LdVs)


# 반올림/소수점 제한 주기
# precision: 명시된 자리 수(int)만큼 소숫점 아래 자리가 출력됨. 디폴트는 8이다. 
# linewidth: The number of characters per line for the purpose of inserting line breaks
# suppress: If True, always print floating point numbers using fixed point notation ex) 1e-4 이런식으로 ㄴㄴ
np.set_printoptions(precision=4, linewidth=250, suppress=True)

# load and define parameters
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load mesh_list
mesh_list = np.load(os.path.join(config['python_data_path'], config['sorted_maya_bs_mesh_list'])).astype(str)
num_blendshapes = len(mesh_list)
print("num_blendshapes", num_blendshapes)

# get LdV
start = time.time()
LdV = build_L_deltaV(mesh_list, config['python_save_path_bsObj'], config['neutral_pose'])
print("Done computing in:", time.time() - start)

# reshape and save
# np.reshape()
# - '-1' : 해당 dim은 자동으로 지정됨 
LdV = np.reshape(LdV, (np.shape(LdV)[0], -1)) # same as Ldv before.. why reshape..?
print("shape LdV", np.shape(LdV))
np.save(os.path.join(config['python_data_path'], config['LdV_name']), LdV)
print("Saved!")
