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


# 求权重的最佳值 （求46期的均值，若单独取某一期的数据会存在季节性变化的差异）
def get_wight_better_para(simuStandLAI, fileDatas, landCover, qualityControl, type):
    # Spatial
    standLAI = np.array(simuStandLAI).mean(axis=0)
    if type == 1:
        winsi_len = 11
        lineAll = []
        for euc_pow in range(1, 6):
            oneLine = []
            for winSize in range(3, winsi_len):
                oneYear = []
                for index in range(0, 46):
                    result = Filling_Pixel.Spatial_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, euc_pow, winSize, position=(0,0))
                    oneYear.append(result)
                yearMean = np.array(oneYear).mean(axis=0)
                calRMSE = math.sqrt((1/(len(fileDatas[0])*len(fileDatas[0][0]))) * np.sum(np.square(standLAI - yearMean)))
                oneLine.append(calRMSE)
            lineAll.append(oneLine)
        
        Draw_PoltLine.draw_polt_Line(np.arange(3, winsi_len, 1),{
            'title': '' ,
            'xlable': 'Half Width',
            'ylable': 'RMSE',
            'line': lineAll,
            'le_name': ['Pow=1', 'Pow=2', 'Pow=3', 'Pow=4', 'Pow=5'],
            'color': False,
            'marker': False,
            'lineStyle': []
            },'./Daily_cache/0126/0126_Spa_%s'% ('all'), False, 1)
            
    # Temporal
    else:
        tem_len = 10
        SES_pow_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        lineAll = []
        for ses_pow in SES_pow_array:
            oneLine = []
            for temporalLength in range(1, tem_len):
                oneYear = []
                for index in range(0, 46):
                    result = Filling_Pixel.Temporal_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, temporalLength, 5, ses_pow, position=(0,0))
                    oneYear.append(result)
                yearMean = np.array(oneYear).mean(axis=0)
                calRMSE = math.sqrt((1/(len(fileDatas[0])*len(fileDatas[0][0]))) * np.sum(np.square(standLAI - yearMean)))
                oneLine.append(calRMSE)
            lineAll.append(oneLine)
        # pos_count = 50
        # Filling_Pos = random_pos(QC_All[fileIndex], 2000, pos_count)
        # # print(len(Filling_Pos))
        # winsi_len = 10
        # line_array = []
        # SES_pow_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # for ses_pow in SES_pow_array:
        #     print(ses_pow)
        #     pow_one_or = []
        #     pow_one_fil = []
        #     pow_one_we = []
        #     for win_size in range(2, winsi_len):
        #         re = Filling_Pixel.Fill_Pixel_One(fileDatas, fileIndex, Filling_Pos, LC_info, QC_All, 6, win_size, ses_pow, 2, 5, 1)
        #         pow_one_or.append(re['Or'])
        #         pow_one_fil.append(re['Fil'])
        #         pow_one_we.append(round(np.mean(re['Weight']), 3))
        #     line_array.append(pow_one_we)
        # # print(line_array)
        #     # result = calculatedif(pow_one_or, pow_one_fil, winsi_len-5, len(Filling_Pos))
        #     # line_array.append(result['RMSE'])
        Draw_PoltLine.draw_polt_Line(np.arange(2, tem_len, 1),{
            'title': '',
            'xlable': 'Half TemLen',
            'ylable': 'RMSE',
            'line': lineAll,
            'le_name': ['Pow=0.2', 'Pow=0.3','Pow=0.4', 'Pow=0.5', 'Pow=0.6', 'Pow=0.7'],
            'color': False,
            'marker': False,
            'lineStyle': []
            },'./Daily_cache/0126/0126_Tem', False, 1)

simuStandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
fileDatas = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')

get_wight_better_para(simuStandLAI[:,100:200,100:200], fileDatas[:,100:200,100:200], LandCover[100:200,100:200], Err_weight[:,100:200,100:200], 2)