import os
from turtle import color
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
    

def drawScatter(x, y, h, v, type):
    calRMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y)))
    parameter = rsquared(x, y)
    print(calRMSE, parameter)
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
    plt.savefig(f'./PNG/h{h}v{v}_{type}', dpi=300)
    plt.show()

def calRaw():  
    h = '12'
    v = '04'
    fileLists = readDir(f'../HDF/h{h}v{v}')
    data = pd.read_csv(f'./Site_Classification/站点_h{h}v{v}.csv', usecols= ['Site value', 'line', 'samp', 'c6 DOY'], dtype={'Site value': float, 'line': float, 'samp': float, 'c6 DOY': str})
    # print(data)
    MODISValue = []
    spatialValue = []
    tmporalValue = []
    for ele in data.values:
        # print(ele)
        file_index = (int(ele[3][4:])  - 1 ) / 8
        raw = Read_HDF.calculate_RawMean(fileLists, int(file_index), int(ele[1]), int(ele[2]))
        spa = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/h{h}v{v}_2018/Spatial/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        tem = Read_HDF.calculate_TemSpaMean(f'../Improved_RealData/h{h}v{v}_2018/Temporal/LAI_{int(file_index) + 1}.npy', int(ele[1]), int(ele[2]))
        MODISValue.append(raw / 10)
        spatialValue.append(spa / 10)
        tmporalValue.append(tem / 10)

    # print(data['Site value'])
    # x = data['Site value']
    # y = MODISValue
    specific = pd.DataFrame({'Site': data['Site value'], 'Raw': MODISValue, 'Spatial':spatialValue, 'Temporal': tmporalValue})
    specific.to_csv(f'./Site_Analysis/h{h}v{v}.csv')


h = '12'
v = '04'
data = pd.read_csv(f'./Site_Analysis/h{h}v{v}.csv', dtype=float)
drawScatter(data['Site'], data['Raw'], h, v, 'Raw')
drawScatter(data['Site'], data['Spatial'], h, v, 'Spatial')
drawScatter(data['Site'], data['Temporal'], h, v, 'Temporal')