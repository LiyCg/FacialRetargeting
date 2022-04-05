import maya.cmds as cmds
import numpy as np
import os
import json

with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# get all blendshapes' meshes
mesh_list = cmds.ls(config['maya_bs_group'], dag=1, type="mesh")  # get all blenshapes from the blenshape group folder

# remove names issue and make a list of string instead of that "maya" list [u"", u""]
# maya 상의 list가 아닌 순수한 이름들 string으로 이루어진 name으로 list만드는 과정
mesh_list_tuple = []
for mesh in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    # create blendshape string list
    # str(mesh[:-remove_letters]) : ShapeOrig 혹은 Shape가 제거된 bshp의 순수한 이름
    mesh_list_tuple.append(str(mesh[:-remove_letters]))

# check if mesh retrieved succesfully
print("mesh_list_tuple")
print(mesh_list_tuple)

# create a blendshape nodes for every blendshape mesh
# cmds.blendShape() : finds last 'string' specified as base
# - mesh_list_tuple : targets
# - config['maya_base_mesh_name'] : base 
# - name : used to sepcify the name of the node being created 
cmds.blendShape(mesh_list_tuple, config['maya_base_mesh_name'], name=config['blendshape_node'])

# save mesh names externally as numpy (single) array to binary file in Numpy format(='.npy')
# os.path.join() : 인수에 전달된 2개의 문자열을 결합하여, 1개의 경로로 할 수 있다 / 새로운 파일을 생성할 수도 있다.
# config['python_data_path'] 여기에다가, config['maya_bs_mesh_list'] 라는 파일이름으로(dir 새로 생성 후), mesh_list_tuple을 저장
np.save(os.path.join(config['python_data_path'], config['maya_bs_mesh_list']), mesh_list_tuple)
