import os
from typing import MappingView
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


# calculate RMSE
def cal_RMSE():
    pos_arr= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Filling_Pos.npy', allow_pickle=True)
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')   
    index = 1
    data_array = np.loadtxt('../Simulation/Filling/2018_%s' % (index+1))
    numera_rmse = 0
    for ele in pos_arr[index]:
        v_rmse = math.pow((LAI_Simu_noErr[index][ele[0]][ele[1]] - LAI_Simu_addErr[index][ele[0]][ele[1]]), 2)
        # v_rmse = math.pow((LAI_Simu_noErr[index][ele[0]][ele[1]] - data_array[ele[0]][ele[1]]), 2)
        numera_rmse += v_rmse
    RMSE = round(math.sqrt(numera_rmse / len(pos_arr[index])), 2)
    print(RMSE)


# calculate R平方
def cal_R_R():
    pos_arr= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Filling_Pos.npy', allow_pickle=True)
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')   
    index = 5
    data_array = np.loadtxt('../Simulation/Filling/2018_%s' % (index+1))
    val_t = 0
    val_b = 0
    for ele in pos_arr[index]:
        # v_rmse = math.pow((LAI_Simu_noErr[index][ele[0]][ele[1]] - LAI_Simu_addErr[index][ele[0]][ele[1]]), 2)
        # t_one = math.pow((data_array[ele[0]][ele[1]] - 2.1), 2)
        t_one = math.pow((LAI_Simu_addErr[index][ele[0]][ele[1]] - 2.1), 2)
        b_one = math.pow((LAI_Simu_noErr[index][ele[0]][ele[1]] - 2.1), 2)
        val_t += t_one
        val_b += b_one
    R_R = round(val_t / val_b, 2)
    print(R_R)

# 找出所有需填补的像元位置
def get_position_filling():
    Err= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Err_peren.npy')
    pos_arr = []
    for day in range(0, 46):
        print(day)
        one = []
        for i in range(0, 500):
            for j in range(0, 500):
                val = Err[day][i][j]
                if val > 0 : one.append([i,j])
        pos_arr.append(one)
    np.save('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Filling_Pos', pos_arr)

# 模拟数据的时空填补(所有像元)
def Simu_filling_All():
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    # LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Err_weight.npy')
    
    pos_arr= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Filling_Pos.npy', allow_pickle=True)
    ses_pow = 0.8
    for index in range(6, 45):
        print(index)
        Filling_Pos = pos_arr[index]
        # Filling_Pos = [[0, 20]]
        # if index < 10: ses_pow = 0.3
        # if 10 < index < 14 or 36 < index < 40: ses_pow = 0.5
        # elif 14 < index < 20 or 30 < index < 36: ses_pow = 0.8
        # else: ses_pow = 0.3
        # elif 20 < index < 30: ses_pow = 0.3
        re1 = Filling_Pixel.Fill_Pixel_All(LAI_Simu_addErr, index, Filling_Pos, LandCover, Err_weight, 6, 12, ses_pow, 2, 5)
        # re1 = Filling_Pixel.Fill_Pixel_noQC(LAI_Simu_addErr, index, Filling_Pos, LandCover, 6, 12, ses_pow, 2, 5)

# 模拟数据的时空填补(单像元)
def Simu_filling(x_v, y_v, idx):
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    # LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Err_weight.npy')
    
    Filling_Pos = [[x_v, y_v]]
    Fil_val_1 = []
    Fil_val_2 = []
    Fil_val = []
    ori_val = []
    simu_val = []

    ses_pow = 0.8
    for index in range(1, 45):
        # if index < 10: ses_pow = 0.3
        # if 10 < index < 14 or 36 < index < 40: ses_pow = 0.5
        # elif 14 < index < 20 or 30 < index < 36: ses_pow = 0.8
        # else: ses_pow = 0.3
        # elif 20 < index < 30: ses_pow = 0.3
        re1 = Filling_Pixel.Fill_Pixel(LAI_Simu_addErr, index, Filling_Pos, LandCover, Err_weight, 6, 12, ses_pow, 2, 5)
        # re1 = Filling_Pixel.Fill_Pixel_noQC(LAI_Simu_addErr, index, Filling_Pos, LandCover, 6, 12, ses_pow, 2, 5)
        Fil_val_1.append(re1['Tem'][0]/10)
        Fil_val_2.append(re1['Spa'][0] /10)
        Fil_val.append(re1['Fil'][0] /10)
        ori_val.append(re1['Or'][0] /10)
        simu_val.append(LAI_Simu_noErr[index][x_v][y_v] / 10)
   
    Draw_PoltLine.draw_polt_Line(np.arange(1, 45, 1),{
        # 'title': 'pos_%s_%s' % (x_v, y_v),
        'title': '',
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [simu_val, ori_val, Fil_val],
        'le_name': ['Real','Trial', 'Filling','Temporal', 'Spatial'],
        'color': ['gray', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
        'marker': False,
        'lineStyle': ['dashed'],
        },'./Daily_cache/0316/Filling_%s' % idx, True, 2)

# Simu_filling_All() 

# Simu_filling(200, 200, 1)

# Simu_filling(362, 24, 2)