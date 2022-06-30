from osgeo import gdal
import numpy as np
import numpy.ma as ma
from pathlib import Path
import os
import time
import sys
sys.path.append("..")
from Filling import Improved_Pixel, ReadDirFiles, Public_Methods


def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC}

sites = {
    'BART': {'h': 12, 'v': 4, 'line': 1424.16, 'samp': 2105.61},
    'BLAN': {'h': 11, 'v': 5, 'line': 225.04, 'samp': 2250.38},
    'CPER': {'h': 10, 'v': 4, 'line': 2203.77, 'samp': 173.89},
    'DSNY': {'h': 10, 'v': 6, 'line': 449.49, 'samp': 1962.62},
    'DELA': {'h': 10, 'v': 5, 'line': 1789.49, 'samp': 1435.02},
    'GUAN': {'h': 11, 'v': 7, 'line': 486.81, 'samp': 1533.85},
    'HARV': {'h': 12, 'v': 4, 'line': 1790.43, 'samp': 1636.72},
    'JERC': {'h': 10, 'v': 5, 'line': 2112.74, 'samp': 1858.18},
    'JORN': {'h': 8, 'v': 5, 'line': 1777.73, 'samp': 2394.90},
    'KONA': {'h': 10, 'v': 5, 'line': 212.99, 'samp': 1207.90},
    'LAJA': {'h': 11, 'v': 7, 'line': 474.40, 'samp': 1490.80},
    'MOAB': {'h': 9, 'v':5, 'line': 419.89, 'samp': 981.96},
    'NIWO': {'h': 9, 'v':4, 'line': 2386.47, 'samp': 2203.54},
    'ONAQ': {'h': 9, 'v':4, 'line': 2356.88, 'samp': 978.91},
    'ORNL': {'h': 11, 'v':5, 'line': 968.11, 'samp': 427.40},
    'OSBS': {'h': 10, 'v':6, 'line': 77.22, 'samp': 2099.01},
    'SCBI': {'h': 11, 'v':5, 'line': 265.20, 'samp': 2203.28},
    'SERC': {'h': 12, 'v':5, 'line': 265.86, 'samp': 97.75},
    'STEI': {'h': 11, 'v':4, 'line': 1077.35, 'samp': 1731.83},
    'STER': {'h': 10, 'v':4, 'line': 2288.63, 'samp': 386.25},
    'SRER': {'h': 8, 'v':5, 'line': 1940.94, 'samp': 1419.03},
    'TALL': {'h': 10, 'v':5, 'line': 1691.39, 'samp': 1599.03},
    'UNDE': {'h': 11, 'v':4, 'line': 903.35, 'samp': 1935.23},
    'WOOD': {'h': 11, 'v':4, 'line': 688.72, 'samp': 594.74},
} # 24个

# 时空相关性计算、权重计算 
for key, ele in sites.items():
    print(key)
    hv = 'h%02dv%02d' % (ele['h'], ele['v'])
    line = int(ele['line'])
    samp = int(ele['samp'])
    site = key
    url = f'./Site_Calculate/{site}'
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')

    if not Path(url).is_dir(): 
        os.mkdir(url)
        dirList = ['Temporal', 'Spatial', 'Temporal_N', 'Spatial_N', 'Temporal_Weight', 'Spatial_Weight', 'Raw_Weight', 'Improved']
        for ele in dirList:
            os.mkdir(f'{url}/{ele}')

    LAIDatas = []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])
    rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
    LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
    LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
    landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()[line-10:line+11, samp-10:samp+11]

    qualityControl = np.load(f'../QC/Version_4/{hv}_2018/{hv}_Weight.npy')[:, line-10:line+11, samp-10:samp+11]

    # 时空相关性计算
    for index in range(0, 46): 
        Tem = Improved_Pixel.Temporal_Cal(rawLAI, index, landCover, qualityControl, 3,  0.5)
        np.save(f'{url}/Temporal/LAI_{index + 1}', Tem)
        Spa = Improved_Pixel.Spatial_Cal(rawLAI, index, landCover, qualityControl, 2,  4)
        np.save(f'{url}/Spatial/LAI_{index + 1}', Spa)
        
    #     # 不含质量控制
    #     Tem_N = Improved_Pixel.Temporal_Cal_N(rawLAI, index, landCover, 3,  0.5)
    #     np.save(f'{url}/Temporal_N/LAI_{index + 1}', Tem_N)
    #     Spa_N = Improved_Pixel.Spatial_Cal_N(rawLAI, index, landCover, 2,  4)
    #     np.save(f'{url}/Spatial_N/LAI_{index + 1}', Spa_N)

    # 权重计算 
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
    # (version_1)
    # for index in range(0, 46):
    #     SpaWeight = Improved_Pixel.Spatial_Weight(rawLAI, spa_LAI, index, qualityControl, 3)
    #     np.save(f'{url}/Spatial_Weight/LAI_{index + 1}', SpaWeight)
    #     TemWeight = Improved_Pixel.Temporal_Weight(rawLAI, tem_LAI, index, qualityControl, landCover, 4)
    #     np.save(f'{url}/Temporal_Weight/LAI_{index + 1}', TemWeight)

    # (version_2)
    for index in range(1, 45):
        rawWeight = Improved_Pixel.cal_TSS(rawLAI, index)
        np.save(f'{url}/Raw_Weight/LAI_{index + 1}', rawWeight)
        spaWeight = Improved_Pixel.cal_TSS(spa_LAI, index)
        np.save(f'{url}/Spatial_Weight/LAI_{index + 1}', spaWeight)
        temWeight = Improved_Pixel.cal_TSS(tem_LAI, index)
        np.save(f'{url}/Temporal_Weight/LAI_{index + 1}', temWeight)


# 加权平均求最终计算值（version_2)
for key, ele in sites.items():
    print(key)
    hv = 'h%02dv%02d' % (ele['h'], ele['v'])
    line = int(ele['line'])
    samp = int(ele['samp'])
    site = key
    url = f'./Site_Calculate/{site}'
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
    
    LAIDatas = []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])
    rawLAI = np.array(LAIDatas)[1:45, line-10:line+11, samp-10:samp+11]
    qualityControl = (np.load(f'../QC/Version_4/{hv}_2018/{hv}_Weight.npy')[1:45, line-10:line+11, samp-10:samp+11])

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
    # 质量小于5的像元不用于最终的权重计算
    # pos = qualityControl < 5
    # raw_Weight[pos] = 0
    # 计算权重求和后的值
    for i in range(46):
        if i == 0 or i == 45:
            np.save(f'{url}/Improved/LAI_{i + 1}', np.load(f'{url}/Temporal/LAI_{i + 1}.npy'))
        else:
            one = (ma.masked_greater(tem_LAI[i-1], 70) * tem_Weight[i-1] + ma.masked_greater(spa_LAI[i-1], 70) * spa_Weight[i-1] + ma.masked_greater(rawLAI[i-1], 70) * raw_Weight[i-1]) / (tem_Weight[i-1] + spa_Weight[i-1] + raw_Weight[i-1])
            pos = rawLAI[i-1].__gt__(70)
            one[pos] = rawLAI[i-1][pos]
            np.save(f'{url}/Improved/LAI_{i + 1}', np.array(one))



# 加权平均求最终计算值（version_1)
for key, ele in sites.items():
    print(key)
    hv = 'h%02dv%02d' % (ele['h'], ele['v'])
    line = int(ele['line'])
    samp = int(ele['samp'])
    site = key
    url = f'./Site_Calculate/{site}'
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
    
    LAIDatas = []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])
    rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
    qualityControl = (np.load(f'../QC/Version_2/{hv}_2018/{hv}_Weight.npy')[:, line-10:line+11, samp-10:samp+11]) / 10

    tem = []
    spa = []
    temWei = []
    spaWei = []
    for i in range(1, 47):
        # print(i)
        spa_data = np.load(f'{url}/Spatial/LAI_{i}.npy')
        tem_data = np.load(f'{url}/Temporal/LAI_{i}.npy')
        spa_wei = np.load(f'{url}/Spatial_Weight/LAI_{i}.npy')
        tem_wei = np.load(f'{url}/Temporal_Weight/LAI_{i}.npy')
        tem.append(tem_data)
        spa.append(spa_data)
        temWei.append(tem_wei)
        spaWei.append(spa_wei)
    tem_LAI = np.array(tem)
    spa_LAI = np.array(spa)
    tem_Weight = np.array(temWei)
    spa_Weight = np.array(spaWei)

    # 归一化
    TemWeight = (tem_Weight - np.min(tem_Weight)) / (np.max(tem_Weight) - np.min(tem_Weight))
    SpaWeight = (spa_Weight - np.min(spa_Weight)) / (np.max(spa_Weight) - np.min(spa_Weight))
    

    # 计算权重求和后的值
    for i in range(46):
        one = (ma.masked_greater(tem_LAI[i], 70) * TemWeight[i] + ma.masked_greater(spa_LAI[i], 70) * SpaWeight[i] + ma.masked_greater(rawLAI[i], 70) * qualityControl[i]) / (TemWeight[i] + SpaWeight[i] + qualityControl[i])
        pos = rawLAI[i].__gt__(70)
        one[pos] = rawLAI[i][pos]
        np.save(f'{url}/Improved/LAI_{i + 1}', np.array(one))

