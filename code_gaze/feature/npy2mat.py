import numpy as np
from scipy import io

mat = np.load('A.npy')
io.savemat('A.mat', {'gene_features': mat})
mat = np.load('D.npy')
io.savemat('D.mat', {'gene_features': mat})
