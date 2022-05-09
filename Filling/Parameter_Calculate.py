import os
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.ma.core import array
from numpy.random.mtrand import sample
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import matplotlib.patches as patches
from matplotlib import animation 
import copy
import ReadDirFiles
import math
import h5py
import time
import random
import Filling_Pixel
import Draw_PoltLine

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

def random_pos(QC, ran_len, length):
    rand_pos_1 = int_random(0, 2399, ran_len)
    rand_pos_2 = int_random(0, 2399, ran_len)
    fill_pos = []
    for ele in range(0, ran_len):
        if QC[rand_pos_1[ele]][rand_pos_2[ele]] == 10:
            fill_pos.append([rand_pos_1[ele], rand_pos_2[ele]])
            if len(fill_pos) == length: return fill_pos


# 求参数的最佳值 （求46期的均值，若单独取某一期的数据会存在季节性变化的差异）
def get_wight_better_para(simuStandLAI, fileDatas, landCover, qualityControl, type):
    # Spatial
    standLAI = np.array(simuStandLAI).mean(axis=0)
    if type == 1:
        winsi_len = 11
        lineAll = []
        for winSize in range(5, winsi_len):
            print(winSize)
            oneLine = []
            for euc_pow in range(1, 6):
                oneYear = []
                for index in range(0, 46):
                    result = Filling_Pixel.Spatial_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, euc_pow, winSize)
                    oneYear.append(result)
                yearMean = np.array(oneYear).mean(axis=0)
                calRMSE = math.sqrt((1/(len(fileDatas[0])*len(fileDatas[0][0]))) * np.sum(np.square(standLAI - yearMean)))
                oneLine.append(calRMSE)
            lineAll.append(oneLine)
                   
    # Temporal
    else:
        tem_len = 8
        SES_pow_array = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        lineAll = []
        for ses_pow in SES_pow_array:
            oneLine = []
            for temporalLength in range(1, tem_len):
                oneYear = []
                for index in range(0, 46):
                    result = Filling_Pixel.Temporal_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, temporalLength, 5, ses_pow)
                    oneYear.append(result)
                yearMean = np.array(oneYear).mean(axis=0)
                calRMSE = math.sqrt((1/(len(fileDatas[0])*len(fileDatas[0][0]))) * np.sum(np.square(standLAI - yearMean)))
                oneLine.append(calRMSE)
            lineAll.append(oneLine)

    return lineAll

simuStandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
fileDatas = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
left = 100
right = 200
paraType = 1
lineAll = get_wight_better_para(simuStandLAI[:,left:right,left:right], fileDatas[:,left:right,left:right], LandCover[left:right,left:right], Err_weight[:,left:right,left:right], paraType)
print(lineAll)
if paraType == 1:
    Draw_PoltLine.draw_polt_Line(np.arange(1, 6, 1),{
        'title': '' ,
        'xlable': 'Correlation Coefficient',
        'ylable': 'RMSE',
        'line': lineAll,
        # 'le_name': ['Pow=1', 'Pow=2', 'Pow=3', 'Pow=4', 'Pow=5'],
        'le_name': ['HW=5','HW=6', 'HW=7','HW=8','HW=8','HW=9','HW=10'],
        'color': False,
        'marker': False,
        'size': False,
        'lineStyle': []
        },'./Daily_cache/0506/Spatial_Para_(100,200)', True, 1)
else:
    Draw_PoltLine.draw_polt_Line(np.arange(1, 8, 1),{ #tem_len
        'title': '',
        'xlable': 'Half Length',
        'ylable': 'RMSE',
        'line': lineAll,
        'le_name': ['cc=0.20', 'cc=0.25','cc=0.30', 'cc=0.35', 'cc=0.40', 'cc=0.45', 'cc=0.50'],
        'color': False,
        'marker': False,
        'size': False,
        'lineStyle': []
        },'./Daily_cache/0506/Temporal_Para_(100,200)', True, 1)