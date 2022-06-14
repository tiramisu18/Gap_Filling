import numpy as np
from osgeo import gdal
import ReadDirFiles
import math
import random
import Filling_Pixel
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

# qualityControl = np.load('../QC/Version_1/%s_2018/%s_AgloPath_Wei.npy' % (tile, tile))
qualityControl = np.load('../QC/Version_3/%s_Weight.npy' % (tile))

# 时空相关性计算
for index in range(0, 46): 
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(index)
    Tem = Filling_Pixel.Temporal_Cal_Matrix_Tile(np.array(LAIDatas), index, landCover, qualityControl, 3,  0.5)
    np.save('./Daily_cache/0614/Temporal1/LAI_%s'% (index + 1), Tem)
    # np.save('../Improved_RealData/%s_2018/Temporal/LAI_%s'% (tile, (index + 1)), Tem)
    # Spa = Filling_Pixel.Spatial_Cal_Matrix_Tile(np.array(LAIDatas), index, landCover, qualityControl, 2,  4)
    # np.save('../Improved_RealData/%s_2018/Spatial/LAI_%s'% (tile, (index + 1)), Spa)
