import numpy as np


if __name__ == '__main__':
    a = np.ones([5,8])
    print(a)
    b = [[0], [1]]
    a[b] = 0
    print(a)
    mask_train = np.where(a == 1)
    print(mask_train)
