from enum import Flag
import os
import numpy as np
import numpy.ma as ma
from numpy.core.fromnumeric import mean
from numpy.ma.core import array
from numpy.random.mtrand import sample
import ReadDirFiles
import math
import h5py
import Filling_Pixel
import Draw_PoltLine
import Public_Motheds
import Newtons_Method


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

# 模拟数据的时空提升
def Simu_improved():
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    # LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
    

    for index in range(0,46): 
        print(index)
        # result = Filling_Pixel.Temporal_Cal_Matrix_Tile(LAI_Simu_addErr, index, LandCover, Err_weight, 3,  0.35)
        # np.savetxt('./Daily_cache/0506/Tem_LAI/LAI_%s'% (index + 1), result)
        # result = Filling_Pixel.Spatial_Cal_Matrix_Tile(LAI_Simu_addErr, index, LandCover, Err_weight, 3,  8)
        # np.savetxt('./Daily_cache/0506/Spa_LAI/LAI_%s'% (index + 1), result)

        # result = Filling_Pixel.Fill_Pixel_Matrix(LAI_Simu_addErr, index, LandCover, Err_weight, 6, 12, ses_pow, 2, 5, position=tuple(Position))
        # Filling_Pixel.Calculate_Weight(result['Tem'], result['Spa'], LAI_Simu_addErr[index], LandCover, Err_weight[index], tuple(Position))
        Filling_Pixel.Calculate_Weight(np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % (index+1))[0:15, 0:15], np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % (index+1))[0:15, 0:15], LAI_Simu_addErr[index, 0:15, 0:15], LandCover[0:15, 0:15], Err_weight[index, 0:15, 0:15], (5,5))
        
        # re1 = Filling_Pixel.Fill_Pixel_noQC(LAI_Simu_addErr, index, Position, LandCover, 6, 12, ses_pow, 2, 5)
        # Fil_val_1.append(re1['Tem'][0]/10)
        # Fil_val_2.append(re1['Spa'][0] /10)
   
    # Draw_PoltLine.draw_polt_Line(np.arange(0, 361, 8),{
    #     # 'title': 'pos_%s_%s' % (x_v, y_v),
    #     'title': '',
    #     'xlable': 'Day',
    #     'ylable': 'LAI',
    #     # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
    #     'line': [LAI_Simu_noErr[:, 5, 5]/10, LAI_Simu_addErr[:, 5, 5]/10],
    #     'le_name': ['Standard','Inaccurate', 'Filling','Temporal', 'Spatial'],
    #     'color': ['gray', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
    #     'marker': False,
    #     'size': False,
    #     'lineStyle': [],
    #     },'./Daily_cache/0506/Improved_(5,5)', True, 2)


# Simu_improved()
# stru_fun= np.load('./Daily_cache/0506/Structure_Lagrange.npy', allow_pickle=True)
# print(stru_fun[20,20])
# Newtons_Method.simulated_martix(stru_fun[20,20])

# Err_peren= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_peren.npy')
# Err_peren= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
# for i in range(0, 10):
#     Public_Motheds.render_Img(Err_peren[i])

# aa = ma.masked_values(np.array(np.arange(0,12)).reshape(2,3,-1), 2)
# bb = np.ones(12).reshape(2,3,-1)
# aa = [{'x': -0.187500000000000, 'y': 1.18750000000000, 'k': 0.0}],[{'x': -0.235294117647059, 'y': 1.23529411764706, 'k': 0.0}]
# aa = [[(-3/16, 19/16, 0), (-1/10, 11/10, 0)],[(-17/22, 39/22, 0), (-5/2, 7/2, 0)]]
# bb = np.array(aa)
# print(bb.shape)
# cc = bb[:,0]
# print(cc)

def Improved_position():
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')

    Spa_Weight = [0,1,0,0,1/2,0,1,0,1,0,2/3,9/8,1,2/3,3/10,5/4,0,1,0,10/29,0,14/29,8/5,229/362,152/103,-47/223,0,-12/17,30/167,81/368,41/153,24/55,20/33,3/7,4/3,8/7,1/2,1,0,2/3,1,0,1/2,1,1/2,0]
    Tem_Weight = [1,0,1,1,1/2,1,0,1,0,1,1/3,-1/8,0,1/3,7/10,-1/4,1,0,1,19/29,1,15/29,-3/5,133/362,-49/103,270/223,1,29/17,137/167,287/368,112/153,31/55,13/33,4/7,-1/3,-1/7,1/2,0,1,1/3,0,1,1/2,0,1/2,1]
    # print(len(Spa_Weight), len(Tem_Weight))

    Spa_improvedArray = []
    Tem_improvedArray = []
    for i in range(1, 47):
        Spa_data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
        Tem_data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
        Spa_improvedArray.append(Spa_data)
        Tem_improvedArray.append(Tem_data)
    Spa_improvedLAI = np.array(Spa_improvedArray)
    Tem_improvedLAI = np.array(Tem_improvedArray)
    Spa_5_5 = Spa_improvedLAI[:,5,5] / 10
    Tem_5_5 = Tem_improvedLAI[:,5,5] / 10
    x = np.array(Spa_Weight)
    y = np.array(Tem_Weight)
    improved_5_5 = (Spa_5_5 * x + Tem_5_5 * y) / (x + y)
    print(improved_5_5)
    # print(Spa_5_5)
    Draw_PoltLine.draw_polt_Line(np.arange(0, 361, 8),{
        # 'title': 'pos_%s_%s' % (x_v, y_v),
        'title': '',
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [LAI_Simu_noErr[:, 5, 5]/10, LAI_Simu_addErr[:, 5, 5]/10, Spa_5_5, Tem_5_5,improved_5_5,  ],
        'le_name': ['Standard','Inaccurate', 'Temporal', 'Spatial', 'Improved',],
        'color': ['#958b8c', '#a4ac80', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c'],
        'marker': False,
        'size': False,
        'lineStyle': ['solid','solid','dashed','dashed','solid',],
        },'./Daily_cache/0506/Improved_(5,5)', True, 2)

def previous_method():
    LAI_Simu_noErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
    pos = [5,5]
    Tem_Lai = []
    Tem_Weight = []
    Spa_Lai = []
    Spa_Weight = []
    for index in range(1, 45):
        print(index)
        Tem = Filling_Pixel.Temporal_Cal(LAI_Simu_addErr, index, pos, LandCover, Err_weight, 3, 8, 0.35)
        Spa = Filling_Pixel.Spatial_Cal(LAI_Simu_addErr, index, pos, LandCover, Err_weight, 3, 8)
        Tem_Lai.append(Tem['filling'])
        Tem_Weight.append(Tem['weight'])
        Spa_Lai.append(Spa['filling'])
        Spa_Weight.append(Spa['filling'])
    
    Spa_5_5 = np.array(Spa_Lai) / 10
    Tem_5_5 =  np.array(Tem_Lai) / 10
    x = np.array(Spa_Weight)
    y = np.array(Tem_Weight)
    improved_5_5 = (Spa_5_5 * x + Tem_5_5 * y) / (x + y)
    Draw_PoltLine.draw_polt_Line(np.arange(8, 353, 8),{
        # 'title': 'pos_%s_%s' % (x_v, y_v),
        'title': '',
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [LAI_Simu_noErr[1:45, 5, 5]/10, LAI_Simu_addErr[1:45, 5, 5]/10, Spa_5_5, Tem_5_5, improved_5_5,],
        'le_name': ['Standard','Inaccurate', 'Temporal', 'Spatial', 'Improved',],
        'color': ['#958b8c', '#a4ac80', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c'],
        'marker': False,
        'size': False,
        'lineStyle': ['solid','solid','dashed','dashed','solid',],
        },'./Daily_cache/0506/Previous_Improved_(5,5)', True, 2)
    