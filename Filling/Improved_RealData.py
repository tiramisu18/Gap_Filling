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

hv = 'h12v04'
fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')

LAIDatas = []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])

# aa = np.array(LAIDatas)
# print(aa[16, 1422:1428, 2103:2109])

LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

qualityControl = np.load(f'../QC/Version_2/{hv}_2018/{hv}_Weight.npy')

# 时空相关性计算
# for index in range(0, 46): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     # Tem = Improved_Pixel.Temporal_Cal(np.array(LAIDatas), index, landCover, qualityControl, 3,  0.5)
#     # np.save(f'../Improved_RealData/{hv}_2018/Temporal/LAI_{index + 1}', Tem)
#     Spa = Improved_Pixel.Spatial_Cal(np.array(LAIDatas), index, landCover, qualityControl, 2,  4)
#     # np.save(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{index + 1}', Spa)

# # 时空相关性计算（不含质量控制）
# for index in range(0, 46): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     # Tem = Improved_Pixel.Temporal_Cal_N(np.array(LAIDatas), index, landCover, 3,  0.5)
#     # np.save(f'../Improved_RealData_N/{hv}_2018/Temporal/LAI_{index + 1}', Tem)
#     Spa = Improved_Pixel.Spatial_Cal_N(np.array(LAIDatas), index, landCover, 2,  4)
#     # np.save(f'../Improved_RealData_N/{hv}_2018/Spatial/LAI_{index + 1}', Spa)


tem = []
spa = []
for i in range(1, 47):
    # print(i)
    spa_data = np.load(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{i}.npy')
    tem_data = np.load(f'../Improved_RealData/{hv}_2018/Temporal/LAI_{i}.npy')
    tem.append(tem_data)
    spa.append(spa_data)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)

# 计算权重
for index in range(18, 46):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(index)
    # SpaWeight = Improved_Pixel.Spatial_Weight(np.array(LAIDatas), spa_LAI, index, qualityControl, 3)
    # np.save(f'../Improved_RealData/{hv}_2018/Spatial_Weight/LAI_{index + 1}', SpaWeight)
    TemWeight = Improved_Pixel.Temporal_Weight(np.array(LAIDatas), tem_LAI, index, qualityControl, landCover, 4)
    # np.save(f'../Improved_RealData/{hv}_2018/Temporal_Weight/LAI_{index + 1}', TemWeight)
    

tem = []
spa = []
for i in range(10, 20):
    # print(i)
    # spa_data = np.load(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{i}.npy')
    tem_data = np.load(f'../Improved_RealData/{hv}_2018/Temporal_Weight/LAI_{i}.npy')
    Public_Methods.render_Img(tem_data)