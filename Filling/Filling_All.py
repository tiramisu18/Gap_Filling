# from pyhdf.SD import SD, SDC

# file_name = 'C:\JR_Program\Filling_Missing_Values\MOD15A2H.A2000049.h26v05.006.2015136143539.hdf'
# file = SD(file_name, SDC.READ) #加载数据

# print(file.info())


import os
from typing import MappingView
import numpy as np
from numpy.random.mtrand import sample
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import matplotlib.patches as patches
import copy
import ReadDirFiles
import math
import h5py
import Filling_m4
import time

fileLists = ReadDirFiles.readDir(
  'C:\JR_Program\Filling_Missing_Values\h27v07_all')
# print('lists', len(fileLists))

def ReadFile(path):
    # mcd_file_path = 'C:\JR_Program\Filling_Missing_Values\MOD15A2H.A2018361.h27v07.006.2019009094143.hdf' 
    # mcd_file_path = 'C:\JR_Program\MOD15A2H.A2018361.h27v07.006.2019009094143.hdf' 
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    # print(subdatasets)
 
    # for sd in subdatasets:
    #     print('Name: {0}\nDescription:{1}\n'.format(*sd))

    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()

    
    # print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # namestr = path.split('\\')[-1].split('.')[1]
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle(namestr)

    # data = [LAI_part, LAIPart_backup]
    # images = []

    # for i in range(2):
    # # Generate data with a range that varies from one plot to the next.
    #     images.append(axs[i].imshow(data[i], cmap= plt.cm.jet))
    #     axs[i].label_outer()

    # plt.axis('off')
    # # plt.imshow(LAI_part, cmap = 'gray')
    # # plt.imshow(LAIPart_backup, cmap= 'gray') #cmap= plt.cm.jet
    
    # # plt.savefig("./h27v07_test_pic/"+ namestr+".png", dpi=300)
    # plt.show()
    return LAI


fileDatas = []
count = 20
# for file in fileLists:
#   fileDatas.append(ReadFile(file))

# index = 41
# oneSample = fileDatas[index]

# random missing position
# missing_position1 = np.random.randint(500,800,count)
# missing_position2 = np.random.randint(200,500,count)
# position_nan = []
# for i in range(count):
#   for j in range(count):   
#     position_nan.append([missing_position1[i], missing_position2[j]])

# print('nan_length', len(position_nan))



LC_file = gdal.Open('C:\JR_Program\Filling_Missing_Values\h27v07_LC\MCD12Q1.A2018001.h27v07.006.2019200013550.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集

LC_info = gdal.Open(LC_subdatasets[0][0]).ReadAsArray()
# plt.imshow(LC_info, cmap = plt.cm.jet)  # cmap= plt.cm.jet
# # colbar = plt.colorbar()
# plt.show()


# Original Image
# colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400']
# bounds = [0,20,40,70,250]
# cmap = pltcolor.ListedColormap(colors)
# norm = pltcolor.BoundaryNorm(bounds, cmap.N)
# plt.imshow(oneSample, cmap=cmap, norm=norm)
# plt.title('Original Image', family='Times New Roman', fontsize=18)
# cbar = plt.colorbar()
# cbar.set_ticklabels(['0','20','40','70', '250'])
# plt.show()


# read MQC file
path='./MQC_files/h2018_1004_MQC_Score.mat'             
MQC_File=h5py.File(path) 
# print(MQC_File.keys())
file_Content = MQC_File["MQC_Score"]

# for k in range(0, 42):
#   ds = MQC_File[MQC_File["MQC_Score"][0,k]]  # [column, row]
#   MQC_Score = ds[:] 
#   data_one = copy.deepcopy(MQC_Score)
#   MQC_Score_30_Pos = []
#   for i in range(2400):
#     for j in range(2400):
#       if MQC_Score[i][j] <= 35: 
#         MQC_Score_30_Pos.append([i, j])
#         data_one[i][j] = 'nan'

#   print(k, len(MQC_Score_30_Pos))
#   plt.imshow(data_one, cmap = plt.cm.jet)  # cmap= plt.cm.jet
#   plt.title('MQC_' + str(k), family='Times New Roman', fontsize=18)
#   colbar = plt.colorbar()
#   plt.show()

# ds = MQC_File[MQC_File["MQC_Score"][0,index-1]]  # [column, row]
# ds = MQC_File[file_Content[0,index - 1]]
# MQC_Score = ds[:]
# MQC_Score_30_Pos = []
# data_one = copy.deepcopy(MQC_Score)
# for i in range(0, 100):
#   for j in range(1700, 1800):
#     if MQC_Score[i][j] <= 35 and oneSample[i][j] <= 70 : 
#       MQC_Score_30_Pos.append([i, j])
#       data_one[i][j] = 'nan'

# print(len(MQC_Score_30_Pos))


# MQC_part = []
# for i in range(0, 200):
#   row = []
#   for j in range(1700, 1900):
#     row.append(MQC_Score[i][j])
#   MQC_part.append(row)

# plt.imshow(MQC_part, cmap = plt.cm.jet)  # cmap= plt.cm.jet

# plt.title('MQC', family='Times New Roman', fontsize=18)
# colbar = plt.colorbar()
# plt.axis('off')
# plt.show()

# Filling_m4.Fill_m4(fileDatas, index, MQC_Score_30_Pos, LC_info, MQC_File)
# Filling_m3.Fill_m3(fileDatas, index, position_nan, LC_info, MQC_File)
# Filling_m4.Fill_m4(fileDatas, index, position_nan, LC_info, MQC_File) 
# Filling_m5.Fill_m5(fileDatas, index, position_nan, LC_info, MQC_File)



# 去除图片空白区域
# fig = plt.gcf()
# fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)
# fig.savefig("./h27v07_test_pic/MQC_1008_2.png", format='png', transparent=True, dpi=300, pad_inches = 0)

# plt.show()


# f = open('./name.txt',mode='w')        
# f.write(data_one)
# f.close()

# for file_index in range(30, 35):
#   print(file_index)
#   ds = MQC_File[file_Content[0,file_index - 1]]
#   MQC_Score = ds[:]
#   oneSample = fileDatas[file_index]
#   MQC_Score_30_Pos = []
#   # data_one = copy.deepcopy(MQC_Score)
#   for i in range(2400):
#     for j in range(2400):
#       if MQC_Score[i][j] <= 35 and oneSample[i][j] <= 70 : 
#         MQC_Score_30_Pos.append([i, j])
#         # data_one[i][j] = 'nan'

#   print(len(MQC_Score_30_Pos))
#   Filling_m4.Fill_m4(fileDatas, file_index, MQC_Score_30_Pos, LC_info, MQC_File)


# render original data
# colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
# bounds = [0,10,20,30,40,50,60,70,250]
# cmap = pltcolor.ListedColormap(colors)
# norm = pltcolor.BoundaryNorm(bounds, cmap.N)

# for file_index in range(45, 46):

#   # plt.title('Filling Image', family='Times New Roman', fontsize=18)
#   plt.imshow(fileDatas[file_index], cmap=cmap, norm=norm)
#   # cbar = plt.colorbar()
#   # cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
# # 去除图片空白区域
#   fig = plt.gcf()
#   fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
#   plt.gca().xaxis.set_major_locator(plt.NullLocator())
#   plt.gca().yaxis.set_major_locator(plt.NullLocator())
#   plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#   plt.margins(0,0)
#   fig.savefig("./result/h27v07_ori_png/h27v07_" + str(file_index + 1) + ".png", format='png', transparent=True, dpi=300, pad_inches = 0)
#   plt.show()





