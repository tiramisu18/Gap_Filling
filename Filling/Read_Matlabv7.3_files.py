import scipy 
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltcolor
import matplotlib.cm as cm
 
import h5py
path='./LAI_h12v09.mat'                    #需要读取的mat文件路径
feature=h5py.File(path)               #读取mat文件
# data = feature['feature_data'][:] 
print(feature.keys())
print(feature['LAI_year'].shape)

# dataFile = './LAI_h12v09.mat'
# data = scio.loadmat(dataFile)
# print(data)