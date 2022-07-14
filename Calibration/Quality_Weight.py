import numpy as np
import numpy.ma as ma
from osgeo import gdal
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from Filling import Improved_Pixel, Public_Methods, ReadDirFiles

# 读取HDF文件数据集
def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    StdLAI = gdal.Open(subdatasets[5][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC, 'StdLAI': StdLAI}

# 权重加上StdLAI和相对TSS
def addStdLAITSS(StdLAIDatas, TSSValues, hv, line, samp, boundaryValue):
    qualityControl = np.load(f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei.npy')[:, line-10:line+11, samp-10:samp+11]
    # 将质量等级为5的备用算法修改为3
    pos = qualityControl == 5
    qualityControl[pos] = boundaryValue
    # StdLAI有效值范围为0-100
    otherMin = boundaryValue * 0.05
    std = ma.masked_greater(StdLAIDatas, 100)
    map_std = 0.5 + ((otherMin - 0.5) / (0.5 + ma.max(std) - ma.min(std))) * (std - ma.min(std)) 
    tss = ma.array(TSSValues, mask = pos)
    # tss中大于1的值直接设为0.15，小于1的值映射到0.16-0.5
    map_tss = 0.5 + ((otherMin + 0.01 - 0.5) / (1 - 0) * (tss - 0)) 
    p2 = map_tss < 0
    map_tss[p2] = otherMin
    control = map_std + map_tss
    surplus = np.array(ma.filled(control, 1))
    final_qualityControl = surplus * qualityControl
    return final_qualityControl


def cal_TSS(LAIDatas, index):
    numerators = np.absolute(((LAIDatas[index + 1] - LAIDatas[index - 1]) * index) - (LAIDatas[index] * 2) - ((LAIDatas[index + 1] - LAIDatas[index - 1]) * (index - 1)) + (LAIDatas[index - 1] * 2))
    denominators = np.sqrt(np.square(LAIDatas[index + 1] - LAIDatas[index - 1]) + 2**2)
    # absoluteTSS = (numerators / denominators) / 10
    absoluteTSS = numerators / denominators
    relativeTSS = np.round(absoluteTSS / LAIDatas[index], 2)
    return np.nan_to_num(relativeTSS, posinf=10, neginf=10)

def calculate_mean(data):
    step1 = data[8:14, 8:14]
    step2 = ma.masked_greater(step1, 70)
    return ma.mean(step2) * 0.1

def draw_Line (y1, y2, y3, y4, gx, gy, RMSE, savePath = '', issave = False, title = ''):
    x = np.arange(0, 361, 8)
    fig = plt.figure(figsize=(12, 6)) #宽，高
    ax = fig.add_subplot()
    plt.title(title, family='Times New Roman', fontsize=18)   
    plt.xlabel('Day', fontsize=15, family='Times New Roman') 
    plt.ylabel('LAI', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    line1=plt.plot(x,y1, label='count', color='gray',  marker='o', markersize=3)
    line2=plt.plot(x,y2, label='count', color='#ffe117',  marker='.', markersize=3, linestyle= 'dashed')
    line3=plt.plot(x,y3, label='count', color='#bfdb39',  marker='^', markersize=3, linestyle= 'dashed')
    line4=plt.plot(x,y4, label='count', color='#fd7400',  marker='+', markersize=3)
    # line6=plt.plot(x,y6, label='count', color='#b8defd',  marker='H', markersize=3)
    line5=plt.scatter(gx,gy, label='count', color='#fd7400')
    ax.text(310, 3, f'{RMSE[0]} (RMSE)',
        verticalalignment='bottom', horizontalalignment='left',
        color='gray', fontsize=18, family='Times New Roman')
    ax.text(310, 2.7, f'{RMSE[1]}',
        verticalalignment='bottom', horizontalalignment='left',
        color='#ffe117', fontsize=18, family='Times New Roman')
    ax.text(310, 2.4, f'{RMSE[2]}',
        verticalalignment='bottom', horizontalalignment='left',
        color='#bfdb39', fontsize=18, family='Times New Roman')
    ax.text(310, 2.1, f'{RMSE[3]}',
        verticalalignment='bottom', horizontalalignment='left',
        color='#fd7400', fontsize=18, family='Times New Roman')
    plt.legend(
    (line1[0],  line2[0],  line3[0], line4[0], line5), 
    ('Raw', 'Temporal', 'Spatial', 'Improved', 'GBOV'),
    loc = 2, prop={'size':15, 'family':'Times New Roman'},
    )
    if issave :plt.savefig(savePath, dpi=300)
    plt.show()

# 单站点从质量设置到计算出最终值的全过程
def oneSiteQC_Weight(hv, site, line, samp):
    # hv = 'h10v05' #'h12v04' 'h11v05'
    # site = 'KONA' #'HARV' 'BART' 'SCBI'
    # line = 212 #1790   1424   265 
    # samp = 1207 #1636   2105   2203 
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
    LAIDatas, StdLAIDatas = [], []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])
        # QCDatas.append(result['QC'])
        StdLAIDatas.append(result['StdLAI'])
    rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
    rawLAI = np.array(rawLAI, dtype=float)
    StdLAIDatas = np.array(StdLAIDatas)[:, line-10:line+11, samp-10:samp+11]

    TSSArray = np.ones((1,rawLAI.shape[1], rawLAI.shape[2])) 
    for index in range(1,45):
        one = cal_TSS(rawLAI, index)
        TSSArray = np.append(TSSArray, one.reshape(1, one.shape[0], one.shape[1]), axis=0)
    TSSArray = np.append(TSSArray, np.ones((1,rawLAI.shape[1], rawLAI.shape[2])), axis=0)

    LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
    LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
    landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()[line-10:line+11, samp-10:samp+11]

    for value in range(3, 5):
        print(value)
        # value = 3 
        qcWeight = addStdLAITSS(StdLAIDatas, TSSArray, hv, line, samp, value)
        # qcWeight = np.load(f'../QC/Version_5/{hv}_2018/{hv}_Weight.npy')[:, line-10:line+11, samp-10:samp+11]
        # return
        rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
        temporalList, spatialList, improvedList = [], [], []
        for index in range(0, 46): 
            Tem = Improved_Pixel.Temporal_Cal(rawLAI, index, landCover, qcWeight, 3,  0.5)
            temporalList.append(Tem)
            Spa = Improved_Pixel.Spatial_Cal(rawLAI, index, landCover, qcWeight, 2,  4)
            spatialList.append(Spa)
        temLAI = np.array(temporalList)
        spaLAI = np.array(spatialList)

        for index in range(0, 46):
            if index == 0 or index == 45:
                improvedList.append(temLAI[index])
            else:
                rawWeight = Improved_Pixel.cal_TSS(rawLAI, index)
                spaWeight = Improved_Pixel.cal_TSS(spaLAI, index)
                temWeight = Improved_Pixel.cal_TSS(temLAI, index)
                one = (ma.masked_greater(temLAI[index], 70) * temWeight + ma.masked_greater(spaLAI[index], 70) * spaWeight + ma.masked_greater(rawLAI[index], 70) * rawWeight) / (temWeight + spaWeight + rawWeight)
                pos = rawLAI[index].__gt__(70)
                one[pos] = rawLAI[index][pos]
                improvedList.append(one)
        improvedLAI = np.array(improvedList)

        data = pd.read_csv(f'../GBOV/Site_Classification/站点_{hv}.csv', usecols= ['Site name', 'Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site name': str, 'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
        specific = data.loc[data['Site name'] == f'{site}'] 

        GBOVDay, GBOVValue, day_i = [], [], []
        MODISValue, spatialValue, temporalValue, improvedValue = [], [], [], []
        for i in range(46):
            MODISValue.append(calculate_mean(rawLAI[i]))
            spatialValue.append(calculate_mean(spaLAI[i]))
            temporalValue.append(calculate_mean(temLAI[i]))
            improvedValue.append(calculate_mean(improvedLAI[i]))
            currentDOY = i * 8 + 1 
            ele = np.array(specific.loc[specific['c6 DOY'] == '2018%03d' % currentDOY]['Site value'])
            if len(ele) > 0:
                GBOVDay.append(currentDOY - 1)
                GBOVValue.append(ele.mean())
                day_i.append(i)

        onlyGBOVDay_raw = np.array(MODISValue)[day_i]
        onlyGBOVDay_spa = np.array(spatialValue)[day_i]
        onlyGBOVDay_tem = np.array(temporalValue)[day_i]
        onlyGBOVDay_imp = np.array(improvedValue)[day_i]
        # print(GBOVValue, onlyGBOVDay_raw, onlyGBOVDay_tem, onlyGBOVDay_imp)
        RMSE_raw = np.round(np.sqrt((1/len(onlyGBOVDay_raw))* np.sum(np.square(np.array(onlyGBOVDay_raw) - np.array(GBOVValue)))), 2)
        RMSE_tem = np.round(np.sqrt((1/len(onlyGBOVDay_tem))* np.sum(np.square(np.array(onlyGBOVDay_tem) - np.array(GBOVValue)))), 2)
        RMSE_spa = np.round(np.sqrt((1/len(onlyGBOVDay_spa))* np.sum(np.square(np.array(onlyGBOVDay_spa) - np.array(GBOVValue)))), 2)
        RMSE_imp = np.round(np.sqrt((1/len(onlyGBOVDay_imp))* np.sum(np.square(np.array(onlyGBOVDay_imp) - np.array(GBOVValue)))), 2)
        print(RMSE_raw, RMSE_tem, RMSE_spa, RMSE_imp)
        draw_Line(MODISValue, temporalValue, spatialValue, improvedValue, GBOVDay, GBOVValue, [RMSE_raw, RMSE_tem, RMSE_spa, RMSE_imp], issave=True, savePath=f'./PNG/{hv}_{site}_Line', title=f'{site}')


# oneSiteQC_Weight('h12v04', 'BART', 1424, 2105)

sites = {
    'BART': {'h': 12, 'v': 4, 'line': 1424.16, 'samp': 2105.61},
    'HARV': {'h': 12, 'v': 4, 'line': 1790.43, 'samp': 1636.72},
    'SCBI': {'h': 11, 'v':5, 'line': 265.20, 'samp': 2203.28},
    'TALL': {'h': 10, 'v':5, 'line': 1691.39, 'samp': 1599.03},
    # 'KONA': {'h': 10, 'v': 5, 'line': 212.99, 'samp': 1207.90},
    # 'ONAQ': {'h': 9, 'v':4, 'line': 2356.88, 'samp': 978.91},  
    # 'SRER': {'h': 8, 'v':5, 'line': 1940.94, 'samp': 1419.03},  
    # 'WOOD': {'h': 11, 'v':4, 'line': 688.72, 'samp': 594.74},
} # 均一植被类型


Raw_RMSEs, Tem_RMSEs, Spa_RMSEs, Imp_RMSEs = [], [], [], []
for value in range(1, 10):
    print(value)
    GBOVValue = []
    MODISValue, spatialValue, temporalValue, improvedValue = [], [], [], []
    for key, ele in sites.items():
        hv = 'h%02dv%02d' % (ele['h'], ele['v'])
        site = key
        line = int(ele['line'])
        samp = int(ele['samp'])

        fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
        LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
        LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
        landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()[line-10:line+11, samp-10:samp+11]

        LAIDatas, StdLAIDatas = [], []
        for file in fileLists:
            result = ReadFile(file)
            LAIDatas.append(result['LAI'])
            # QCDatas.append(result['QC'])
            StdLAIDatas.append(result['StdLAI'])
        rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
        rawLAI = np.array(rawLAI, dtype=float)
        StdLAIDatas = np.array(StdLAIDatas)[:, line-10:line+11, samp-10:samp+11]

        TSSArray = np.ones((1,rawLAI.shape[1], rawLAI.shape[2])) 
        for index in range(1,45):
            one = cal_TSS(rawLAI, index)
            TSSArray = np.append(TSSArray, one.reshape(1, one.shape[0], one.shape[1]), axis=0)
        TSSArray = np.append(TSSArray, np.ones((1,rawLAI.shape[1], rawLAI.shape[2])), axis=0)

        qcWeight = addStdLAITSS(StdLAIDatas, TSSArray, hv, line, samp, value)

        rawLAI = np.array(LAIDatas)[:, line-10:line+11, samp-10:samp+11]
        temporalList, spatialList, improvedList = [], [], []
        for index in range(0, 46): 
            Tem = Improved_Pixel.Temporal_Cal(rawLAI, index, landCover, qcWeight, 3,  0.5)
            temporalList.append(Tem)
            Spa = Improved_Pixel.Spatial_Cal(rawLAI, index, landCover, qcWeight, 2,  4)
            spatialList.append(Spa)
        temLAI = np.array(temporalList)
        spaLAI = np.array(spatialList)


        for index in range(0, 46):
            if index == 0 or index == 45:
                improvedList.append(temLAI[index])
            else:
                rawWeight = Improved_Pixel.cal_TSS(rawLAI, index)
                spaWeight = Improved_Pixel.cal_TSS(spaLAI, index)
                temWeight = Improved_Pixel.cal_TSS(temLAI, index)
                one = (ma.masked_greater(temLAI[index], 70) * temWeight + ma.masked_greater(spaLAI[index], 70) * spaWeight + ma.masked_greater(rawLAI[index], 70) * rawWeight) / (temWeight + spaWeight + rawWeight)
                pos = rawLAI[index].__gt__(70)
                one[pos] = rawLAI[index][pos]
                improvedList.append(one)
        improvedLAI = np.array(improvedList)

        data = pd.read_csv(f'../GBOV/Site_Classification/站点_{hv}.csv', usecols= ['Site name', 'Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site name': str, 'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
        specific = data.loc[data['Site name'] == f'{site}'] 

        for i in range(46):            
            currentDOY = i * 8 + 1 
            ele = np.array(specific.loc[specific['c6 DOY'] == '2018%03d' % currentDOY]['Site value'])
            if len(ele) > 0:
                GBOVValue.append(ele.mean())
                MODISValue.append(calculate_mean(rawLAI[i]))
                spatialValue.append(calculate_mean(spaLAI[i]))
                temporalValue.append(calculate_mean(temLAI[i]))
                improvedValue.append(calculate_mean(improvedLAI[i]))

    RMSE_raw = np.round(np.sqrt((1/len(MODISValue))* np.sum(np.square(np.array(MODISValue) - np.array(GBOVValue)))), 2)
    RMSE_tem = np.round(np.sqrt((1/len(temporalValue))* np.sum(np.square(np.array(temporalValue) - np.array(GBOVValue)))), 2)
    RMSE_spa = np.round(np.sqrt((1/len(spatialValue))* np.sum(np.square(np.array(spatialValue) - np.array(GBOVValue)))), 2)
    RMSE_imp = np.round(np.sqrt((1/len(improvedValue))* np.sum(np.square(np.array(improvedValue) - np.array(GBOVValue)))), 2)
        
    Raw_RMSEs.append(RMSE_raw)
    Tem_RMSEs.append(RMSE_tem)
    Spa_RMSEs.append(RMSE_spa) 
    Imp_RMSEs.append(RMSE_imp)

np.save('./QC_RMSE(4_v1)', {'Raw_RMSEs':Raw_RMSEs, 'Tem_RMSEs':Tem_RMSEs, 'Spa_RMSEs': Spa_RMSEs, 'Imp_RMSEs':Imp_RMSEs})
# RMSEValues = np.load('./QC_RMSE(BART & HARV & SCBI & TALL).npy', allow_pickle=True).item()
# print(RMSEValues)
print(Raw_RMSEs, Tem_RMSEs, Spa_RMSEs, Imp_RMSEs)
Public_Methods.draw_polt_Line(np.arange(1, 10),{
    'title': '',
    'xlable': 'weight',
    'ylable': 'RMSE',
    'line': [Raw_RMSEs, Tem_RMSEs, Spa_RMSEs, Imp_RMSEs],
    'le_name': ['Raw', 'Temporal', 'Spatial', 'Improved'],
    'color': ['gray', '#bfdb39', '#ffe117', '#fd7400'],
    'marker': [',', 'o', '^', '*'],
    'size': {'width': 10, 'height': 6},
    'lineStyle': ['solid', 'dashed', 'dashed']
    },f'./PNG/all_qc_weight', True, 1)