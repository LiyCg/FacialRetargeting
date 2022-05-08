import numpy as np


def get_soft_mask(dskm):
    """
    compute soft max vector uk
    || dskm || denotes the displacement x/y/z of marker m of blendshape k in delta space
    d: delta
    s: sparse character
    k: blendshapes
    m: markers

    k:= number of blendshapes
    m:= number of markers

    :param dskm: delta_skm (k, m, xyz)
    :return:
    """
    # compute norm
    # axis=2 즉, channel, 여기서는 k, m, 3 순이니깐, 
    # output = k개 만큼의, 각 m별 norm값(scalar)를 m개원소로 갖는 row벡터를, 쌓아놓은 행렬
    norm_dskm = np.linalg.norm(dskm, axis=2) # k X m 
    # get max norm
    # amax() output = row(sk) 별로 max값 찾아서 row vector로 반환 (,k)
    # expand_dims() output = (k X 1)
    # repeat() output = k X m
    max_norm = np.repeat(np.expand_dims(np.amax(norm_dskm, axis=1), axis=1), np.shape(dskm)[1], axis=1) # k X m
    # compute soft max
    return np.reshape(np.repeat(norm_dskm / max_norm, 3), (np.shape(dskm)[0], np.shape(dskm)[1]*np.shape(dskm)[2]))


if __name__ == '__main__':
    """
    test get_soft_max function
    
    run: python -m src.get_soft_max
    """
    np.random.seed(0)
    # declare variables
    n_k = 4 # bshp 갯수
    n_m = 2 # marker 갯
    dsk = np.random.rand(n_k, n_m, 3)  # (k, m, xyz)수

    # visualize dsk
    for i in range(len(dsk)):
        print("dsk: ", dsk[i])

    # build ukm control using double loops
    ukm_control = np.zeros((n_k, n_m, 3))

    for k in range(n_k):
        # compute max norm
        max_norm = 0
        for m in range(n_m):
            norm_dskm = np.linalg.norm(dsk[k, m])
            if norm_dskm > max_norm:
                max_norm = norm_dskm
            print("max_norm", max_norm)
        # compute ukm
        for m in range(n_m):
            norm_dskm = np.linalg.norm(dsk[k, m])
            # 이런식으로 인위적으로 채워 넣는것! 'max_norm 대비 각 marker들의 norm의 비율' 을 각 marker들의 xyz에 채워넣은 것이 softmask
            ukm_control[k, m, 0] = norm_dskm / max_norm
            ukm_control[k, m, 1] = norm_dskm / max_norm
            ukm_control[k, m, 2] = norm_dskm / max_norm

    ukm_control = np.reshape(ukm_control, (n_k, n_m*3))

    # test compute_corr_coef with 2 dims array
    ukm = get_soft_mask(dsk)

    print("ukm", np.shape(ukm))
    print(ukm)
    print("ukm_control", np.shape(ukm_control))
    print(ukm_control)

    # np.around(): Evenly round to the given number of decimals. 소숫점 아래 n째자리로 반올림
    # np.all() : Returns True if all elements evaluate to True.
    assert (np.around(ukm, 6).all() == np.around(ukm_control, 6).all())
    print("get_soft_max function works!")
