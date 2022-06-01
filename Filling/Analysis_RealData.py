import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy.ma as ma
from osgeo import gdal
import Public_Methods
import ReadDirFiles

def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    return {'LAI': LAI}

# 计算absolute TSS，index范围（1，45）
def cal_TSS(LAIDatas, index, landCover, lcType):
    # LAI_maArray = ma.array(ma.masked_greater(LAIDatas[index, ...], 70), mask=landCover.__ne__(lcType))
    # Public_Methods.render_LAI(LAI_maArray)
    numerators = np.absolute(((LAIDatas[index + 1] - LAIDatas[index - 1]) * index) - (LAIDatas[index] * 2) - ((LAIDatas[index + 1] - LAIDatas[index - 1]) * (index - 1)) + (LAIDatas[index - 1] * 2))
    denominators = np.sqrt(np.square(LAIDatas[index + 1] - LAIDatas[index - 1]) + 2**2)
    absoluteTSS = (numerators / denominators) / 10
    return absoluteTSS

# 计算数据提升之后不同植被类型像元平均年季曲线变化
def landCover_Improved(raw, spatial, temporal, landCover, lcType):    
    raw_mean = []
    spa_mean= []
    tem_mean = []
    for i in range(0, 46):
        raw_ma = ma.array((ma.masked_greater(raw[i], 70)), mask=(landCover != lcType))
        spa_ma = ma.array((ma.masked_greater(spatial[i], 70)), mask=(landCover != lcType))
        tem_ma = ma.array((ma.masked_greater(temporal[i], 70)), mask=(landCover != lcType))
        raw_mean.append(np.mean(raw_ma)/10)
        spa_mean.append(np.mean(spa_ma)/10)
        tem_mean.append(np.mean(tem_ma)/10)

    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        'title': 'B%s'% lcType ,
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [raw_mean, spa_mean, tem_mean],
        'le_name': ['Raw','Spatial', 'Temporal'],
        'color': ['#bfdb39', 'gray', '#fd7400'],
        'marker': ['o', ',', '^', '.' ],
        'size': {'width': 10, 'height': 6},
        'lineStyle': ['solid', 'dashed']
        },'./Daily_cache/0530/lc_type_%s'% lcType, True, 2)

lcType = 4
tile = 'h12v03'
# LC
LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h12v03.006.2019199205320.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

# Raw LAI
fileLists = ReadDirFiles.readDir('../HDF/%s' % tile)
lai = []
for file in fileLists:
    result = ReadFile(file)
    lai.append(result['LAI'])
raw_LAI = np.array(lai, dtype=float)

# Temporal Spatial LAI
tem = []
spa = []
for i in range(1, 47):
    print(i)
    spa_data = np.loadtxt('../Imporved_RealData/%s_2018/Spatial/LAI_%s' % (tile, i))
    tem_data = np.loadtxt('../Imporved_RealData/%s_2018/Temporal/LAI_%s' % (tile, i))
    tem.append(tem_data)
    spa.append(spa_data)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)
    
landCover_Improved(raw_LAI, spa_LAI, tem_LAI, landCover, lcType)

# raw_TSS = []
# spa_TSS = []
# tem_TSS = []
# for i in range(1,45):
#     print(i)
#     raw_one = cal_TSS(raw_LAI, i, landCover, lcType)
#     spa_one = cal_TSS(spa_LAI, i, landCover, lcType)
#     tem_one = cal_TSS(tem_LAI, i, landCover, lcType)
#     # print(one.shape)
#     raw_one_ma = ma.array(ma.array(raw_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
#     spa_one_ma = ma.array(ma.array(spa_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
#     tem_one_ma = ma.array(ma.array(tem_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
#     # Public_Methods.render_Img(,issave=True, savepath='./Daily_cache/0530/test%s'% i)
#     raw_TSS.append(raw_one_ma)
#     spa_TSS.append(spa_one_ma)
#     tem_TSS.append(tem_one_ma)

# Public_Methods.render_Img((ma.array(raw_TSS)).sum(axis=0),issave=True, savepath='./Daily_cache/0530/TSS_Raw')
# Public_Methods.render_Img((ma.array(spa_TSS)).sum(axis=0),issave=True, savepath='./Daily_cache/0530/TSS_Spa')
# Public_Methods.render_Img((ma.array(tem_TSS)).sum(axis=0),issave=True, savepath='./Daily_cache/0530/TSS_Tem')
