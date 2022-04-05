import maya.cmds as cmds
import json

# python의 파일 스트림 관련 기능
# : file을 open하면 반드시 close 해줘야하는데 'with~as~:' 를 사용하면 해당 구문이 끝나면 자동으로 닫히므로 실수 줄일 수 있다. 
# 경로는 해당 파일을 저장해둔 경로로 설정을해서 열도록한다. 
# as f : open()한 파일을 f라고 지칭하겠다는 의미
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    #json.load(f)
    # : json 파일을 dict type으로 읽어들이기
    config = json.load(f)
    # with 구분 끝났으니 파일은 close 된다. 

# triangulate base mesh
# config의 key 'maya_base_mesh_name' 에 해당하는 value값을 저장
cmds.polyTriangulate(config['maya_base_mesh_name'])

# get all blendshapes' meshes
mesh_list = cmds.ls(config['maya_bs_group'], dag=1, type="mesh")  # get all blenshapes from blenshape group

# triangualte each blendshape mesh
# blenshape 이름만 따내기 위해서, mesh_list로부터 파이썬 내장 문자열함수를 사용한다 
for mesh in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    
    # mesh 'string'에서 뒤에서부터 remove_letters 만큼은 제외하고 
    # 그 전까지만 포함한 'string'만 가져와 str으로 mesh_name에 저장. 
    mesh_name = str(mesh[:-remove_letters])
    
    # triangulate mesh
    cmds.polyTriangulate(mesh_name)

    # delete history
    cmds.delete(mesh_name, ch=True)
