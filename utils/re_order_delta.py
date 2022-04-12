import numpy as np


def re_order_delta(data):
    """
    re-order delta matrix data according to its norm
    the re-ordering goes from the bigger to smallest norm difference

    :param data:
    :return:
    """
    # compute total displacement
    if len(np.shape(data)) == 2:
        d = np.linalg.norm(data, axis=1)
    elif len(np.shape(data)) == 3:
        d = np.linalg.norm(data, axis=(1, 2))
    else:
        raise ValueError("[RE-ORDER DELTA] Data dimensions is not supported!", len(np.shape(data)))

    # get sorted index from bigger to smaller
    # np.argsort() : Perform an indirect sort along the given axis
    # It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
    sorted_index = np.flip(np.argsort(d))

    # sort data
    # list 예를 들면, (0,2,1) 를 np.array에 명시하면, 그 index대로 순서가 바뀜..;; 개쩌는 기능이다. 
    return data[sorted_index], sorted_index


if __name__ == '__main__':
    """
    test re_order function
    
    run: python -m utils.re_order_delta
    """
    np.random.seed(0)

    # test compute trust values
    delta_sk = np.random.rand(6, 2, 3)  # (k, m)
    print("delta_sk", np.shape(delta_sk))
    print(delta_sk)

    sorted_delta_sk, sorted_index = re_order_delta(delta_sk)
    print("shape sorted delta_sk", np.shape(sorted_delta_sk))
    print(sorted_delta_sk)

    print("sorted_idx")
    print(sorted_index)
    list = np.array(["a", "b", "c", "d", "e", "f"])
    print("list")
    print(list)
    sorted_list = list[sorted_index]
    print("sorted_list")
    print(sorted_list)
