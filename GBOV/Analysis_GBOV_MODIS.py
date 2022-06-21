import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import Read_HDF

def readDir(dirPath):
    if dirPath[-1] == '/':
        print('path can not end with /')
        return
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        fileList.sort()
        # print(fileList[17])
        for f in fileList:
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles #合并当前目录与子目录的所有文件路径
            else:
                if f.find('.hdf') != -1:allFiles.append(f)
        print('allFiles', len(allFiles))
        return allFiles
    else:
        return 'error, not a dir'

def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    #a、b、r
    # print(slope, intercept,"r", r_value,"r-squared", r_value**2)
    return [slope, intercept, r_value**2]
    

def drawScatter(x, y, hv, type):
    calRMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y)))
    parameter = rsquared(x, y)
    print('RMSE, a, b, R2', calRMSE, parameter)
    y2 = parameter[0] * x + parameter[1]

    plt.scatter(x, y, color='#bfdb39')
    plt.ylabel(f'{type} LAI', fontsize=15, family='Times New Roman')
    plt.xlabel('GBOV LAI', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    # parameter = np.polyfit(x, y, deg=1)
    # print(parameter)
    
    plt.plot(x, y2, color='#ffe117', linewidth=1, alpha=1)
    plt.plot((0, 7), (0, 7),  ls='--',c='k', alpha=0.8, label="1:1 line")
    plt.savefig(f'./PNG/Scatter/{hv}_{type}', dpi=300)
    plt.show()

def draw_Line (y1, y2, y3, gx, gy, savePath = '', issave = False, title = ''):
    x = np.arange(0, 361, 8)
    plt.figure(figsize=(12, 6)) #宽，高
    plt.title(title, family='Times New Roman', fontsize=18)   
    plt.xlabel('Day', fontsize=15, family='Times New Roman') 
    plt.ylabel('LAI', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    line1=plt.plot(x,y1, label='count', color='gray',  marker='o', markersize=3, linestyle= 'dashed')
    line2=plt.plot(x,y2, label='count', color='#ffe117',  marker='.', markersize=3)
    line3=plt.plot(x,y3, label='count', color='#bfdb39',  marker='^', markersize=3)
    line4=plt.scatter(gx,gy, label='count', color='#fd7400')
    plt.legend(
    (line1[0],  line2[0],  line3[0],  line4), 
    ('Raw', 'Temporal', 'Spatial', 'GBOV'),
    loc = 2, prop={'size':15, 'family':'Times New Roman'},
    )
    if issave :plt.savefig(savePath, dpi=300)
    plt.show()

# Step1: 按照站点_hv.csv计算对应的MODIS位置的LAI值保存为分析数据文件(从计算后的整个Tile中读取)
def calMean_Analysis_Tile():  
    hv = 'h12v04'
    fileLists = readDir(f'../HDF/{hv}')
    data = pd.read_csv(f'./Site_Classification/站点_{hv}.csv', usecols= ['Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
    
    MODISValue = []
    spatialValue = []
    temporalValue = []
    spatialValue_N = []
    temporalValue_N = []
    for ele in data.values:
        # print(ele)
        file_index = (int(ele[3][4:])  - 1 ) / 8
        raw = Read_HDF.calculate_RawMean(fileLists, int(file_index), int(ele[1]), int(ele[2]))
        spa = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        tem = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/{hv}_2018/Temporal/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        spa_n = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData_N/{hv}_2018/Spatial/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        tem_n = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData_N/{hv}_2018/Temporal/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        MODISValue.append(raw / 10)
        spatialValue.append(spa / 10)
        temporalValue.append(tem / 10)
        spatialValue_N.append(spa_n / 10)
        temporalValue_N.append(tem_n / 10)

    # print(data['Site value'])
    # x = data['Site value']
    # y = MODISValue
    specific = pd.DataFrame({'Site': data['Site value'], 'Raw': MODISValue, 'Spatial':spatialValue, 'Temporal': temporalValue, 'Spatial_N': spatialValue_N, 'Temporal_N': temporalValue_N})
    specific.to_csv(f'./Site_Analysis/{hv}.csv')

# Step1: 按照站点_hv.csv计算对应的MODIS位置的LAI值保存为分析数据文件(从站点位置21*21范围中读取)
def calMean_Analysis(hv='h12v04'):  
    # hv = 'h08v05'
    fileLists = readDir(f'../HDF/{hv}')
    data = pd.read_csv(f'./Site_Classification/站点_{hv}.csv', usecols= ['Site name','Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site name': str, 'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})

    MODISValue = []
    spatialValue = []
    temporalValue = []
    spatialValue_N = []
    temporalValue_N = []
    improvedValue = []
    for ele in data.values:
        # print(ele)
        file_index = (int(ele[4][4:])  - 1 ) / 8
        raw = Read_HDF.calculate_RawMean(fileLists, int(file_index), int(ele[2]), int(ele[3]))
        spa = Read_HDF.calculate_part(f'./Site_Calculate/{ele[0]}/Spatial/LAI_{int(file_index) + 1}.npy')
        tem = Read_HDF.calculate_part(f'./Site_Calculate/{ele[0]}/Temporal/LAI_{int(file_index) + 1}.npy')
        spa_n = Read_HDF.calculate_part(f'./Site_Calculate/{ele[0]}/Spatial_N/LAI_{int(file_index) + 1}.npy')
        tem_n = Read_HDF.calculate_part(f'./Site_Calculate/{ele[0]}/Temporal_N/LAI_{int(file_index) + 1}.npy')
        improved = Read_HDF.calculate_part(f'./Site_Calculate/{ele[0]}/Improved/LAI_{int(file_index) + 1}.npy')
        MODISValue.append(raw / 10)
        spatialValue.append(spa / 10)
        temporalValue.append(tem / 10)
        spatialValue_N.append(spa_n / 10)
        temporalValue_N.append(tem_n / 10)
        improvedValue.append(improved / 10)

    specific = pd.DataFrame({'Site': data['Site value'], 'Raw': MODISValue, 'Spatial':spatialValue, 'Temporal': temporalValue, 'Spatial_N': spatialValue_N, 'Temporal_N': temporalValue_N, 'Improved': improvedValue})
    # print(specific)
    specific.to_csv(f'./Site_Analysis/{hv}.csv')

# 合并Site_Analysis单独的Tile
def siteAnalysisMerge():
    data_=pd.DataFrame()
    for inputfile in os.listdir('./Site_Analysis'):
        data_1 = pd.read_csv(f'./Site_Analysis/{inputfile}')
        # data_=data_.append(data_1,ignore_index=True) 
        data_=pd.concat([data_1,data_])

    data_.to_csv(f'./Site_Analysis/All.csv')

# 读取分析数据文件绘制特定站点折线图
def getSiteLine(hv = 'h12v04', site = 'BART'):
    # hv = 'h12v04'
    # site = 'BART'
    data = pd.read_csv(f'./Site_Classification/站点_{hv}.csv', usecols= ['Site name', 'Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site name': str, 'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
    specific = data.loc[data['Site name'] == f'{site}'] 
    fileLists = readDir(f'../HDF/{hv}')
    MODISValue = []
    spatialValue = []
    temporalValue = []
    GBOVDay = []
    GBOVValue = []
    if len(specific) > 0:
        line = int(specific.iloc[0, 2])
        samp =  int(specific.iloc[0, 3])
        for i in range(46):
            # print(i)
            raw = Read_HDF.calculate_RawMean(fileLists, i, line, samp)
            spa = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{i + 1}.npy', line, samp)
            tem = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/{hv}_2018/Temporal/LAI_{i + 1}.npy', line, samp)
            # spa = Read_HDF.calculate_part(f'./Site_Calculate/{site}/Spatial/LAI_{i + 1}.npy')
            # tem = Read_HDF.calculate_part(f'./Site_Calculate/{site}/Temporal/LAI_{i + 1}.npy')
            MODISValue.append(raw / 10)
            spatialValue.append(spa / 10)
            temporalValue.append(tem / 10)
            currentDOY = i * 8 + 1       
            ele = np.array(specific.loc[specific['c6 DOY'] == '2018%03d' % currentDOY]['Site value'])
            if len(ele) > 0:
                GBOVDay.append(currentDOY - 1)
                GBOVValue.append(ele.mean())

    draw_Line(MODISValue, temporalValue, spatialValue, GBOVDay, GBOVValue, issave=True, savePath=f'./PNG/{hv}_{site}_Line', title=f'{site}')

# 读取分析数据文件绘制单个Tile散点图
def getScatterPanel():
    hv = 'all'
    data = pd.read_csv(f'./Site_Analysis/all.csv', dtype=float)
    drawScatter(data['Site'], data['Raw'], hv, 'Raw')
    drawScatter(data['Site'], data['Spatial'], hv, 'Spatial')
    drawScatter(data['Site'], data['Temporal'], hv, 'Temporal')
    drawScatter(data['Site'], (data['Temporal'] + data['Spatial']) / 2, hv, 'Temporal+Spatial')
    drawScatter(data['Site'], data['Spatial_N'], hv, 'Spatial_N')
    drawScatter(data['Site'], data['Temporal_N'], hv, 'Temporal_N')
    drawScatter(data['Site'], (data['Temporal_N'] + data['Spatial_N']) / 2, hv, 'Temporal_N+Spatial_N')
    drawScatter(data['Site'], data['Improved'], hv, 'Improved')



hvLists = ['h08v05', 'h09v04', 'h09v05', 'h10v04', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h11v07', 'h12v04', 'h12v05']

