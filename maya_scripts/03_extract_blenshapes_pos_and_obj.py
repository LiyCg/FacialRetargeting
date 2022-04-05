import maya.cmds as cmds
# numpy는 python에서 수치 데이터를 다루는 가장 basic하고, strong 패키지이다.
import numpy as np
import json
import os

# load parameters for the scenes
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load mesh_list as np array
mesh_list = np.load(os.path.join(config['python_data_path'], config['maya_bs_mesh_list']+'.npy')).astype(str)
print("mesh list")
print(mesh_list)

# vertices which were selected corresponding to actor's marker position
# astype(int) : converting values in config file to int
vtx_list = np.array(config['vrts_pos']).astype(int)
print("num_markers:", len(vtx_list))

# get positions of all the markers across each blendshapes
# 각 blendshape마다 vertexID 어떻게 되어있나 확인해보자 (in Maya) 
bs_vrts_pos = []
for mesh in mesh_list:
    print("mesh", mesh)
    vrts_pos = []
    #각 bshp mesh 별로 iterate하는 loop
    for vtx in vtx_list:
        # cmds.xform() : can be used query/set any element in a transformation node
        # current xform() outputs [linear linear linear] tuple, which is position in worldSpace
        vrts_pos.append(cmds.xform(mesh+".pnts["+str(vtx)+"]", query=True,
                                  translation=True,
                                  worldSpace=True))
    bs_vrts_pos.append(vrts_pos)

    # select and save object
    cmds.select(mesh)
    # pr(preserveReferences) - impot/export시 태그가 붙지 않도록 하는 태그인듯
    ## : modifies the various import/export flags
    ## such that references are imported/exported as actual references rather than copies of those references. 
    # typ(type)
    ## : set the type of this file 
    # es(exportSelected)
    ## : Export the selected items into the specified file
    ## returns the name of the exported file.
    # op(options)
    ## : Set/query the currently set file options. 
    ## file options are used while saving a maya file. Two file option flags supported in current file command are v and p.
    ## v(verbose) indicates whether long or short attribute names and command flags names are used when saving the file. 
    ## Used by both maya ascii and maya binary file formats. It only can be 0 or 1.
    ## Setting v=1 indicates that long attribute names and command flag names will be used. 
    ## By default, or by setting v=0, short attribute names will be used.
    ## p(precision) defines the maya file IO's precision when saving the file. Only used by maya ascii file format.
    ## It is an integer value. The default value is 17.
    ## The option format is "flag1=XXX;flag2=XXX".Maya uses the last v and p as the final result.
    ## Note:
    ## 1. Use a semicolon(";") to separate several flags. 2. No blank space(" ") in option string.
    
    cmds.file(os.path.join(config['python_save_path_bsObj'], mesh +".obj"), pr=1,
              typ="OBJexport",
              es=1,
              # 이 flag 부분은 뭘 의미하는지 나중에 각 bshp들이 어떻게 저장되는지 직접 확인해보면 알 수 있을듯 
              op="groups=0; ptgroups=0; materials=0; smoothing=0; normals=0;")

print("done processing vertices for (n_blendshapes, n_markers, pos):", np.shape(bs_vrts_pos))

# save vertices positions
np.save(os.path.join(config['python_data_path'], config['vertices_pos_name']), bs_vrts_pos)
