import numpy as np
import stim
import pymatching
from scipy.sparse import csc_matrix

n =10
row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i + 1) % n)))
data = np.ones(2 * n, dtype=np.uint8)
print(csc_matrix((data, (row_ind, col_ind))))
