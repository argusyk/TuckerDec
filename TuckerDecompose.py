import numpy as np
import tensorflow as tf
from scipy.io.matlab import loadmat
from ktensor import KruskalTensor

# Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

# Build ktensor and learn CP decomposition using ALS with specified optimizer
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=20000)

# Save reconstructed tensor to file
np.save('X_predict.npy', X_predict)