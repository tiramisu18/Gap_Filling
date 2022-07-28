import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import Read_HDF
import sys
sys.path.append('..')
from Filling import Public_Methods

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
    # print(slope, intercept,'r', r_value,'r-squared', r_value**2)
    return [slope, intercept, r_value**2]
    
def drawScatter(x, y, hv, type, colors):
    calRMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y)))
    parameter = rsquared(x, y)
    print('RMSE, a, b, R2', calRMSE, parameter)
    y2 = parameter[0] * x + parameter[1]
    fig, ax = plt.subplots(figsize=(6,4))
    # ax.scatter(x, y, c=colors, s=20, marker='^')
    ax.scatter(x, y, c=colors, s=20, marker='*')
    plt.ylabel(f'{type} LAI', fontsize=15, family='Times New Roman')
    plt.xlabel('GBOV LAI', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    plt.plot(x, y2, color='#fede89', linewidth=1, alpha=1)    
    plt.plot((0, 7), (0, 7),  ls='--',c='k', alpha=0.8, linewidth=0.8, label='1:1')
    ax.text(0.5, 5.7, r'${R^2}$ = 'f'{round(parameter[2],2)}',
        verticalalignment='bottom', horizontalalignment='left',
        color='#000000', fontsize=14, family='Times New Roman')
    ax.text(0.5, 4.9, r'${RMSE}$ = 'f'{round(calRMSE,2)}',
        verticalalignment='bottom', horizontalalignment='left',
        color='#000000', fontsize=14, family='Times New Roman')
    ax.text(4.5, 2, f'Y = {round(parameter[0],2)}X + {round(parameter[1],2)}',
        verticalalignment='bottom', horizontalalignment='left',
        color='gray', fontsize=12, family='Times New Roman')
    # plt.legend(loc = 2, framealpha=0, labelspacing=1,
    #     prop={'size':20, 'family':'Times New Roman'})
    ax.set(xlim=(0, 7), ylim=(0, 7))
    plt.savefig(f'./PNG/Scatter/{hv}_{type}', dpi=300)
    plt.show()

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
    siteInfo = pd.read_csv(f'./Site_Info.csv', usecols= ['Site','BiomeTypeMark'])    
    MODISValue, spatialValue, temporalValue, improvedValue = [], [], [], []
    spatialValue_N, temporalValue_N = [], []
    biomeType = []
    for ele in data.values:
        # print(ele)
        specific = siteInfo.loc[siteInfo['Site'] == ele[0]] 
        biomeType.append(specific['BiomeTypeMark'].values[0])
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

    specific = pd.DataFrame({'Site': data['Site value'], 'Raw': MODISValue, 'Spatial':spatialValue, 'Temporal': temporalValue, 'Spatial_N': spatialValue_N, 'Temporal_N': temporalValue_N, 'Improved': improvedValue, 'BiomeType': biomeType})
    # print(specific)
    specific.to_csv(f'./Site_Analysis/{hv}.csv')

# Step2: 合并Site_Analysis单独的Tile
def siteAnalysisMerge():
    data_=pd.DataFrame()
    for inputfile in os.listdir('./Site_Analysis'):
        data_1 = pd.read_csv(f'./Site_Analysis/{inputfile}')
        # data_=data_.append(data_1,ignore_index=True) 
        data_=pd.concat([data_1,data_])

    data_.to_csv(f'./Site_Analysis/All.csv')

# 读取分析数据文件绘制特定站点折线图
def getSiteLine(hv = 'h12v04', site = 'BART'):
    data = pd.read_csv(f'./Site_Classification/站点_{hv}.csv', usecols= ['Site name', 'Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site name': str, 'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
    specific = data.loc[data['Site name'] == f'{site}'] 
    fileLists = readDir(f'../HDF/{hv}')
    MODISValue, spatialValue, temporalValue, improvedValue = [], [], [], []
    GBOVDay, GBOVValue = [], []
    onlyGBOVDay_raw, onlyGBOVDay_spa, onlyGBOVDay_tem, onlyGBOVDay_imp = [], [], [], []
    diffValue = []
    if len(specific) > 0:
        line = int(specific.iloc[0, 2])
        samp =  int(specific.iloc[0, 3])
        for i in range(46):
            # print(i)
            raw = Read_HDF.calculate_RawMean(fileLists, i, line, samp)
            spa = Read_HDF.calculate_part(f'./Site_Calculate/{site}/Spatial/LAI_{i + 1}.npy')
            tem = Read_HDF.calculate_part(f'./Site_Calculate/{site}/Temporal/LAI_{i + 1}.npy')
            improved = Read_HDF.calculate_part(f'./Site_Calculate/{site}/Improved/LAI_{i + 1}.npy')
            MODISValue.append(raw / 10)
            spatialValue.append(spa / 10)
            temporalValue.append(tem / 10)
            improvedValue.append(improved / 10)
            currentDOY = i * 8 + 1       
            ele = np.array(specific.loc[specific['c6 DOY'] == '2018%03d' % currentDOY]['Site value'])
            if len(ele) > 0:
                GBOVDay.append(currentDOY - 1)
                GBOVValue.append(ele.mean())
                onlyGBOVDay_raw.append(raw / 10)
                onlyGBOVDay_spa.append(spa / 10)
                onlyGBOVDay_tem.append(tem / 10)
                onlyGBOVDay_imp.append(improved / 10)
                # diffValue.append((f'Day {currentDOY}', np.abs(np.array([raw/10, tem/10, spa/10, ((tem+spa)/2)/10, improved/10]) - ele.mean())))
                diffValue.append((f'Day {currentDOY}', np.abs(np.array([raw/10, tem/10, spa/10, improved/10]) - ele.mean())) if len(diffValue) == 0 else (f'{currentDOY}', np.abs(np.array([raw/10, tem/10, spa/10, improved/10]) - ele.mean())))
    # averageValue = (np.array(spatialValue) + np.array(temporalValue)) / 2

    RMSE_raw = np.round(np.sqrt((1/len(onlyGBOVDay_raw))* np.sum(np.square(np.array(onlyGBOVDay_raw) - np.array(GBOVValue)))), 2)
    RMSE_tem = np.round(np.sqrt((1/len(onlyGBOVDay_tem))* np.sum(np.square(np.array(onlyGBOVDay_tem) - np.array(GBOVValue)))), 2)
    RMSE_spa = np.round(np.sqrt((1/len(onlyGBOVDay_spa))* np.sum(np.square(np.array(onlyGBOVDay_spa) - np.array(GBOVValue)))), 2)
    RMSE_imp = np.round(np.sqrt((1/len(onlyGBOVDay_imp))* np.sum(np.square(np.array(onlyGBOVDay_imp) - np.array(GBOVValue)))), 2)
    print(RMSE_raw, RMSE_tem, RMSE_spa, RMSE_imp)
    draw_Line(MODISValue, temporalValue, spatialValue, improvedValue, GBOVDay, GBOVValue, [RMSE_raw, RMSE_tem, RMSE_spa, RMSE_imp], issave=True, savePath=f'./PNG/Line_Survey/{hv}_{site}_Line', title=f'{site}')

    # category_names = ['Raw', 'Temporal', 'Spatial', 'Tem+Spa', 'Improved']
    category_names = ['Raw', 'Temporal', 'Spatial', 'Improved']
    Public_Methods.survey(dict(diffValue), category_names)
    plt.savefig(f'./PNG/Line_Survey/{hv}_{site}_Line_Survey', dpi=300)
    plt.show()

# 读取分析数据文件绘制单个Tile散点图
def getScatterPanel():
    hv = 'all'
    data = pd.read_csv(f'./Site_Analysis/all.csv', dtype={'Site': float, 'Raw': float, 'Spatial': float, 'Temporal': float, 'Spatial_N': float, 'Temporal_N': float, 'Improved': float, 'BiomeType': int})
    colorsPara = {1:'#e85c46', 2:'#53aead', 3:'#bee5a0', 4: '#fdbe6f'}
    colors = [colorsPara[bioType] for bioType in data['BiomeType']]
    drawScatter(data['Site'], data['Raw'], hv, 'Raw', colors=colors)
    drawScatter(data['Site'], data['Spatial'], hv, 'Spatial', colors=colors)
    drawScatter(data['Site'], data['Temporal'], hv, 'Temporal', colors=colors)
    drawScatter(data['Site'], (data['Temporal'] + data['Spatial']) / 2, hv, 'Temporal+Spatial')
    drawScatter(data['Site'], data['Spatial_N'], hv, 'Spatial_N')
    drawScatter(data['Site'], data['Temporal_N'], hv, 'Temporal_N')
    drawScatter(data['Site'], (data['Temporal_N'] + data['Spatial_N']) / 2, hv, 'Temporal_N+Spatial_N')
    drawScatter(data['Site'], data['Improved'], hv, 'Improved', colors=colors)

# 按照生物群落类型计算RMSE、R2并绘图
def biomeType_cal():
    data = pd.read_csv(f'./Site_Analysis/all.csv', usecols= ['Site','Raw', 'Improved', 'BiomeType'],dtype={'Site': float, 'Raw': float, 'Improved': float, 'BiomeType': int})
    raw_R2s, raw_RMSEs = [], []
    imp_R2s, imp_RMSEs = [], []
    for i in range(1, 5):
        specific = data.loc[data['BiomeType'] == i] 
        x = specific['Site']
        y1 = specific['Raw']
        y2 = specific['Improved']
        y1_RMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y1)))
        y2_RMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y2)))
        y1_r2 = rsquared(x, y1)
        y2_r2 = rsquared(x, y2)
        raw_R2s.append(round(y1_r2[2], 2))
        imp_R2s.append(round(y2_r2[2], 2))
        raw_RMSEs.append(round(y1_RMSE, 2))
        imp_RMSEs.append(round(y2_RMSE, 2))
    
    print(raw_R2s)
    print(imp_R2s)
    print(raw_RMSEs)
    print(imp_RMSEs)
    # plt.style.use('fivethirtyeight')
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6,4))
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    colors = ['#e85c46', '#53aead', '#bee5a0', '#fdbe6f']
    ax.scatter(raw_R2s, raw_RMSEs, c=colors, s=200, marker='^')
    ax.scatter(imp_R2s, imp_RMSEs, c=colors, s=300, marker='*')
    plt.ylabel('RMSE', fontsize=15, family='Times New Roman')
    plt.xlabel('R^2', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    ax.set(xlim=(0.55, 0.95), ylim=(0.5, 1.1), yticks=np.arange(0.5, 1.1, 0.1), xticks=np.arange(0.6, 1, 0.05))
    plt.savefig(f'./PNG/Scatter/biomeType', dpi=300)   
    plt.show()

# 绘制所有站点的密度分布直方图
def getHistDensity(hv = 'h12v04', site = 'BART'):
    hv = 'all'
    data = pd.read_csv(f'./Site_Analysis/All.csv', dtype=float)

    raw = data['Raw'] - data['Site']
    spa = data['Spatial'] - data['Site']
    spa_n = data['Spatial_N'] - data['Site']
    tem = data['Temporal'] - data['Site']
    tem_n = data['Temporal_N'] - data['Site']
    ave = (data['Spatial'] + data['Temporal']) / 2 - data['Site']
    imp = data['Improved'] - data['Site']
    # 绘制误差的分布密度直方图
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(raw, density=True, histtype='stepfilled', bins=50, alpha=1, label='Raw', color='#e44f35')
    ax.hist(tem, density=True, histtype='stepfilled', bins=50, alpha=0.9, label='Temporal', color='#b8defd')
    ax.hist(tem_n, density=True, histtype='stepfilled', bins=50, alpha=0.8, label='Temporal_N', color='#4e6ef2')
    ax.hist(spa, density=True, histtype='stepfilled', bins=50, alpha=0.7, label='Spatial', color='#b3df72')
    ax.hist(spa_n, density=True, histtype='stepfilled', bins=50, alpha=0.7, label='Spatial_N', color='#3faa5a')
    ax.hist(ave, density=True, histtype='stepfilled', bins=50, alpha=0.7, label='Spa+Tem', color='#fdffbe')
    ax.hist(imp, density=True, histtype='stepfilled', bins=50, alpha=0.5, label='Improved', color='#fd7400')

        
    ax.set_xlabel('Absolute Difference', fontsize=15, family='Times New Roman')
    ax.set_ylabel('Density', fontsize=15, family='Times New Roman')
    ax.legend(prop={'size':15, 'family':'Times New Roman'})
    fig.tight_layout()
    # ax.set(xlim=(0, 500))
            # ylim=(0, 8), yticks=np.arange(1, 8))
    plt.xticks( family='Times New Roman', fontsize=15)
    plt.yticks( family='Times New Roman', fontsize=15)
    plt.savefig('./PNG/histogram', dpi=300)
    plt.show()
    
   
# getSiteLine(hv = 'h12v04', site = 'BART')

# hvLists = ['h08v05', 'h09v04', 'h09v05', 'h10v04', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h11v07', 'h12v04', 'h12v05']
# for hv in hvLists:
#     calMean_Analysis(hv=hv)

