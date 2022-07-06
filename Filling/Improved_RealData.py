from re import T
import numpy as np
import numpy.ma as ma
from osgeo import gdal
from sympy import public
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
url = f'../Improved_RealData/{hv}_2018'
LAIDatas = []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])
    # LAIDatas.append(ma.mean(ma.masked_greater(result['LAI'][1450:1650, 1300:1500], 70)))

rawLAI = np.array(LAIDatas)
# print(aa[16, 1422:1428, 2103:2109])
LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()


qualityControl = np.load(f'../QC/Version_4/{hv}_2018/{hv}_Weight.npy')

# 时空相关性计算
# for index in range(0, 46): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     Tem = Improved_Pixel.Temporal_Cal(rawLAI, index, landCover, qualityControl, 3,  0.5)
#     np.save(f'{url}/Temporal/LAI_{index + 1}', Tem)
#     Spa = Improved_Pixel.Spatial_Cal(rawLAI, index, landCover, qualityControl, 2,  4)
#     np.save(f'{url}/Spatial/LAI_{index + 1}', Spa)

# 时空相关性计算（不含质量控制）
# for index in range(0, 10): 
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(index)
#     Tem = Improved_Pixel.Temporal_Cal_N(rawLAI, index, landCover, 3,  0.5)
#     np.save(f'../Improved_RealData/{hv}_2018/Part/Temporal_N/LAI_{index + 1}', Tem)
#     Spa = Improved_Pixel.Spatial_Cal_N(rawLAI, index, landCover, 2,  4)
#     np.save(f'../Improved_RealData/{hv}_2018/Part/Spatial_N/LAI_{index + 1}', Spa)

# print('weight')
# 计算权重
tem = []
spa = []
for i in range(1, 47):
    # print(i)
    spa_data = np.load(f'{url}/Spatial/LAI_{i}.npy')
    tem_data = np.load(f'{url}/Temporal/LAI_{i}.npy')
    tem.append(tem_data)
    spa.append(spa_data)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)

Public_Methods.render_LAI(rawLAI[23])
Public_Methods.render_LAI(tem_LAI[23])
Public_Methods.render_LAI(spa_LAI[23])

# for index in range(1, 45):
#     print(index)
#     rawWeight = Improved_Pixel.cal_TSS(rawLAI, index)
#     np.save(f'{url}/Raw_Weight/LAI_{index + 1}', rawWeight)
#     spaWeight = Improved_Pixel.cal_TSS(spa_LAI, index)
#     np.save(f'{url}/Spatial_Weight/LAI_{index + 1}', spaWeight)
#     temWeight = Improved_Pixel.cal_TSS(tem_LAI, index)
#     np.save(f'{url}/Temporal_Weight/LAI_{index + 1}', temWeight)

print('Improved')
# 加权平均求最终计算值
tem = []
spa = []
temWei = []
spaWei = []
rawWei = []
for i in range(2, 46):
    # print(i)
    spa_data = np.load(f'{url}/Spatial/LAI_{i}.npy')
    tem_data = np.load(f'{url}/Temporal/LAI_{i}.npy')
    spa_wei = np.load(f'{url}/Spatial_Weight/LAI_{i}.npy')
    tem_wei = np.load(f'{url}/Temporal_Weight/LAI_{i}.npy')
    raw_wei = np.load(f'{url}/Raw_Weight/LAI_{i}.npy')
    tem.append(tem_data)
    spa.append(spa_data)
    temWei.append(tem_wei)
    spaWei.append(spa_wei)
    rawWei.append(raw_wei)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)
tem_Weight = np.array(temWei)
spa_Weight = np.array(spaWei)
raw_Weight = np.array(rawWei)
# Public_Methods.render_Img(spa_Weight[0])

# 计算权重求和后的值
for i in range(46):
    print(i)
    if i == 0 or i == 45:
        np.save(f'{url}/Improved/LAI_{i + 1}', np.load(f'{url}/Temporal/LAI_{i + 1}.npy'))
    else:
        one = (ma.masked_greater(tem_LAI[i-1], 70) * tem_Weight[i-1] + ma.masked_greater(spa_LAI[i-1], 70) * spa_Weight[i-1] + ma.masked_greater(rawLAI[i-1], 70) * raw_Weight[i-1]) / (tem_Weight[i-1] + spa_Weight[i-1] + raw_Weight[i-1])
        pos = rawLAI[i-1].__gt__(70)
        one[pos] = rawLAI[i-1][pos]
        np.save(f'{url}/Improved/LAI_{i + 1}', np.array(one))