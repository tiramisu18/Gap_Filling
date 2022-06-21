from re import T
import numpy as np
import numpy.ma as ma
from osgeo import gdal
import ReadDirFiles
import math
import random
import time
import Improved_Pixel
import Public_Methods

def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC}

# 生成一个包含n个介于a和b之间随机整数的数组（不重复）
def int_random(a, b, n) :
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        if(d_int not in a_list) :
            a_list.append(d_int)
        else :
            pass
    return a_list

tile = 'h12v04'
fileLists = ReadDirFiles.readDir('../HDF/%s' % tile)

LAIDatas = []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])

# aa = np.array(LAIDatas)
# print(aa[16, 1422:1428, 2103:2109])

LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h12v04.006.2019199202045.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

qualityControl = np.load('../QC/Version_2/%s_2018/%s_Weight.npy' % (tile, tile))

# 时空相关性计算
# for index in range(0, 46): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     # Tem = Improved_Pixel.Temporal_Cal(np.array(LAIDatas), index, landCover, qualityControl, 3,  0.5)
#     # np.save('../Improved_RealData/%s_2018/Temporal/LAI_%s'% (tile, (index + 1)), Tem)
#     Spa = Improved_Pixel.Spatial_Cal(np.array(LAIDatas), index, landCover, qualityControl, 2,  4)
#     # np.save('../Improved_RealData/%s_2018/Spatial/LAI_%s'% (tile, (index + 1)), Spa)

# # 时空相关性计算（不含质量控制）
# for index in range(0, 46): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     # Tem = Improved_Pixel.Temporal_Cal_N(np.array(LAIDatas), index, landCover, 3,  0.5)
#     # np.save('../Improved_RealData_N/%s_2018/Temporal/LAI_%s'% (tile, (index + 1)), Tem)
#     Spa = Improved_Pixel.Spatial_Cal_N(np.array(LAIDatas), index, landCover, 2,  4)
#     # np.save('../Improved_RealData_N/%s_2018/Spatial/LAI_%s'% (tile, (index + 1)), Spa)


tem = []
spa = []
for i in range(1, 47):
    # print(i)
    spa_data = np.load('../Improved_RealData/%s_2018/Spatial/LAI_%s.npy' % (tile, i))
    tem_data = np.load('../Improved_RealData/%s_2018/Temporal/LAI_%s.npy' % (tile, i))
    tem.append(tem_data)
    spa.append(spa_data)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)

# 计算权重
for index in range(18, 46):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(index)
    # SpaWeight = Improved_Pixel.Spatial_Weight(np.array(LAIDatas), spa_LAI, index, qualityControl, 3)
    # np.save('../Improved_RealData/%s_2018/Spatial_Weight/LAI_%s'% (tile, (index + 1)), SpaWeight)
    TemWeight = Improved_Pixel.Temporal_Weight(np.array(LAIDatas), tem_LAI, index, qualityControl, landCover, 4)
    # np.save('../Improved_RealData/%s_2018/Temporal_Weight/LAI_%s'% (tile, (index + 1)), TemWeight)
    
