import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import copy
import numpy as np

path='/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/MQC/MQC_mat/h27v07_2018_MQC_Score.mat'             
MQC_File=h5py.File(path) 
# print(MQC_File.keys())
file_Content = MQC_File["MQC_Score"]

index = 1
# ds = MQC_File[MQC_File["MQC_Score"][0,index-1]]  # [column, row]
ds = MQC_File[file_Content[0,index]]
MQC_Score = ds[:]


plt.imshow(MQC_Score, cmap = plt.cm.jet)
# 去除图片空白区域
fig = plt.gcf()
fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
fig.savefig("/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/MQC/PNG/MQC_2.png", format='png', transparent=True, dpi=300, pad_inches = 0)

plt.show()