import numpy as np
import torch

# path = 'resources/data/motion_data.npy'
# data = np.load(path)
# print(data.shape)
# data[:, 12] = 0
# data = data[0:16,:]
# np.save('resources/data/motion_data1.npy', data)

path = 'resources/data/motion_data1.npy'
data = np.load(path)
print(data.shape)
ls = list(range(0,27))
ls.remove(17)
ls.remove(18)
ls.remove(19)
ls.remove(24)
ls.remove(25)
ls.remove(26)
data = data[:,ls]
print(data.shape)
np.save('resources/data/motion_data2.npy', data)