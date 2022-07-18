from re import T
import numpy as np
import numpy.ma as ma
from osgeo import gdal
from pathlib import Path
import os
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

hvLists = ['h23v04', 'h29v11', 'h25v06', 'h12v03', 'h11v09', 'h12v04', 'h20v02']

for hv in hvLists:
    print(hv)
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
    url = f'../Improved/Improved_RealData/{hv}_2018'
    LAIDatas = []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])

    rawLAI = np.array(LAIDatas)
    LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
    LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
    landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()
    qualityControl = np.load(f'../QC/Version_5/{hv}_2018/{hv}_Weight.npy')

    dirList = ['Temporal', 'Spatial', 'Temporal_Weight', 'Spatial_Weight', 'Raw_Weight', 'Improved']       
    if not Path(url).is_dir(): 
        os.mkdir(url)
        for ele in dirList:
            os.mkdir(f'{url}/{ele}')
    else:
        for ele in dirList:
            if not Path(f'{url}/{ele}').is_dir(): os.mkdir(f'{url}/{ele}')
    
    # 时空相关性计算
    tem_list, spa_list = [], []
    for index in range(46): 
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(index)
        Tem = Improved_Pixel.Temporal_Cal(rawLAI, index, landCover, qualityControl, 3,  0.5)
        np.save(f'{url}/Temporal/LAI_{index + 1}', Tem)
        Spa = Improved_Pixel.Spatial_Cal(rawLAI, index, landCover, qualityControl, 2,  4)
        np.save(f'{url}/Spatial/LAI_{index + 1}', Spa)
        tem_list.append(Tem)
        spa_list.append(Spa)
    temLAI = np.array(tem_list)
    spaLAI = np.array(spa_list)

    # 时空相关性计算（不含质量控制）
    # for index in range(0, 10): 
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #     print(index)
    #     Tem = Improved_Pixel.Temporal_Cal_N(rawLAI, index, landCover, 3,  0.5)
    #     np.save(f'{url}/Temporal_N/LAI_{index + 1}', Tem)
    #     Spa = Improved_Pixel.Spatial_Cal_N(rawLAI, index, landCover, 2,  4)
    #     np.save(f'{url}/Spatial_N/LAI_{index + 1}', Spa)

    # tem_list, spa_list = [], []
    # for i in range(1, 47):
    #     # print(i)
    #     spa_data = np.load(f'{url}/Spatial/LAI_{i}.npy')
    #     tem_data = np.load(f'{url}/Temporal/LAI_{i}.npy')
    #     tem_list.append(tem_data)
    #     spa_list.append(spa_data)
    # temLAI = np.array(tem_list)
    # spaLAI = np.array(spa_list)

    # 权重计算 + 加权平均得到最终值   
    for index in range(46):
        if index == 0 or index == 45:
            one = (ma.masked_greater(temLAI[index], 70) + ma.masked_greater(spaLAI[index], 70)) / 2
            pos = rawLAI[index].__gt__(70)
            one[pos] = rawLAI[index][pos]
            np.save(f'{url}/Improved/LAI_{index + 1}', np.array(one))
        else:
            rawWeight = Improved_Pixel.cal_TSS(rawLAI, index)
            temWeight = Improved_Pixel.cal_TSS(temLAI, index)
            spaWeight = Improved_Pixel.cal_TSS(spaLAI, index)
            np.save(f'{url}/Raw_Weight/LAI_{index + 1}', rawWeight)
            np.save(f'{url}/Spatial_Weight/LAI_{index + 1}', spaWeight)            
            np.save(f'{url}/Temporal_Weight/LAI_{index + 1}', temWeight)
            one = (ma.masked_greater(temLAI[index], 70) * temWeight + ma.masked_greater(spaLAI[index], 70) * spaWeight + ma.masked_greater(rawLAI[index], 70) * rawWeight) / (temWeight + spaWeight + rawWeight)
            pos = rawLAI[index].__gt__(70)
            one[pos] = rawLAI[index][pos]
            np.save(f'{url}/Improved/LAI_{index + 1}', np.array(one))
