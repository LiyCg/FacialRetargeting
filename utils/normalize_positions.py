import numpy as np

"""
- Numpy 행렬, 벡터 이해

numpy는 사실 column/row vector차이는 없으나 column vector로 표현한다고 보면 쉽다. 
w, wT 가 있을 때, np.dot(wT,w)가 아니라 np.dot(w,wT)를 해야 squared vector가 된다.(Reversed operand position)
행렬도 출력되는 형상을 그대로 보지 말고, transpose 시켜서 이해해야한다. 
array([[0,1],
        [2,3]]) 의 경우, 
        실제로는, 
        0 2
        1 3 인 행렬로 봐야한다!
"""

def normalize_positions(pos, min_pos=None, max_pos=None, return_min=False, return_max=False):
    if min_pos is None:
        # amin(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
        # : return the minimum of an array or mimimum along an axis
        # expand_dims(a, axis)
        # : 명시한 axis로 새로운 axis를 삽입 / 차원 확장
        # repeat(a, repeats, axis=None)
        # : repeat elements of an array, 인풋 a의 axis로 repeats번만큼 반복
        
        # 즉, amin으로 pos 행렬의 행별로 가장 작은 값을 골라서 column vector를 구성한다. 
        # 구성한 벡터의 행에 차원을 확장하여 행렬로 만든다. 
        min_pos = np.repeat(np.expand_dims(np.amin(pos, axis=0), axis=0), np.shape(pos)[0], axis=0)

    pos -= min_pos

    if max_pos is None:
        # max_sk = np.repeat(np.expand_dims(np.amax(ref_sk, axis=0), axis=0), np.shape(ref_sk)[0], axis=0)
        max_pos = np.amax(pos)  # normalize only by the max to keep ratio # todo check if any difference?

    pos /= max_pos

    if return_min and return_max:
        return pos, min_pos, max_pos
    elif return_min:
        return pos, min_pos
    elif return_max:
        return pos, max_pos
    else:
        return pos
