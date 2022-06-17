from enum import Flag
import os
import numpy as np
import numpy.ma as ma
import ReadDirFiles
import math
import Improved_Pixel
import Public_Methods
import Public_Methods
import time

# 模拟数据的时空提升
def Simu_improved():
    LAI_Standard = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    # LAI_Inaccurate = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
    LAI_Inaccurate = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Inaccurate(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')

    for index in range(0, 1): 
        # print('1', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(index)
        # Tem = Improved_Pixel.Temporal_Cal_Matrix_Tile(LAI_Inaccurate, index, LandCover, Err_weight, 3,  0.5)
        # np.savetxt('./Daily_cache/0522/Tem_LAI/LAI_%s'% (index + 1), Tem)
        # Spa = Improved_Pixel.Spatial_Cal_Matrix_Tile(LAI_Inaccurate, index, LandCover, Err_weight, 2,  4)
        # np.savetxt('./Daily_cache/0522/Spa_LAI/LAI_%s'% (index + 1), Spa)


        Improved_Pixel.Fill_Pixel_One(LAI_Inaccurate, index, [337,133], LandCover,  Err_weight, 3, 12, 0.5, 2, 4, 2)
        # result = Improved_Pixel.Fill_Pixel_Matrix(LAI_Inaccurate, index, LandCover, Err_weight, 6, 12, ses_pow, 2, 5, position=tuple(Position))
        # Improved_Pixel.Calculate_Weight(result['Tem'], result['Spa'], LAI_Inaccurate[index], LandCover, Err_weight[index], tuple(Position))
        # Improved_Pixel.Calculate_Weight(np.loadtxt('./Daily_cache/0522/Tem_LAI/LAI_%s' % (index+1))[0:10, 0:10], np.loadtxt('./Daily_cache/0522/Spa_LAI/LAI_%s' % (index+1))[0:10, 0:10], LAI_Inaccurate[index, 0:10, 0:10], LandCover[0:10, 0:10], Err_weight[index, 0:10, 0:10], (5,5))
        
        # re1 = Improved_Pixel.Fill_Pixel_noQC(LAI_Inaccurate, index, Position, LandCover, 6, 12, ses_pow, 2, 5)
        # Fil_val_1.append(re1['Tem'][0]/10)
   
    # Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
    #     # 'title': 'pos_%s_%s' % (x_v, y_v),
    #     'title': '',
    #     'xlable': 'Day',
    #     'ylable': 'LAI',
    #     # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
    #     'line': [LAI_Standard[:, 5, 5]/10, LAI_Inaccurate[:, 5, 5]/10],
    #     'le_name': ['Standard','Inaccurate', 'Filling','Temporal', 'Spatial'],
    #     'color': ['gray', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
    #     'marker': False,
    #     'size': False,
    #     'lineStyle': [],
    #     },'./Daily_cache/0506/Improved_(5,5)', True, 2)


# Simu_improved()


# 计算一个点提升后的LAI值
def Improved_position():
    LAI_Standard = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    LAI_Inaccurate = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')

    # Spa_Weight = [0,1,0,0,1/2,0,1,0,1,0,2/3,9/8,1,2/3,3/10,5/4,0,1,0,10/29,0,14/29,8/5,229/362,152/103,-47/223,0,-12/17,30/167,81/368,41/153,24/55,20/33,3/7,4/3,8/7,1/2,1,0,2/3,1,0,1/2,1,1/2,0]
    # Tem_Weight = [1,0,1,1,1/2,1,0,1,0,1,1/3,-1/8,0,1/3,7/10,-1/4,1,0,1,19/29,1,15/29,-3/5,133/362,-49/103,270/223,1,29/17,137/167,287/368,112/153,31/55,13/33,4/7,-1/3,-1/7,1/2,0,1,1/3,0,1,1/2,0,1/2,1]
    # print(len(Spa_Weight), len(Tem_Weight))

    Spa_improvedArray = []
    Tem_improvedArray = []
    for i in range(1, 47):
        Spa_data = np.loadtxt('./Daily_cache/0522/Spa_LAI/LAI_%s' % i)
        Tem_data = np.loadtxt('./Daily_cache/0522/Tem_LAI/LAI_%s' % i)
        Spa_improvedArray.append(Spa_data)
        Tem_improvedArray.append(Tem_data)
    Spa_improvedLAI = np.array(Spa_improvedArray)
    Tem_improvedLAI = np.array(Tem_improvedArray)
    Spa_5_5 = Spa_improvedLAI[:, 350, 150] / 10
    Tem_5_5 = Tem_improvedLAI[:, 350, 150] / 10
    # x = np.array(Spa_Weight)
    # y = np.array(Tem_Weight)
    # improved_5_5 = (Spa_5_5 * x + Tem_5_5 * y) / (x + y)
    # print(improved_5_5)

    # inaccurate = np.sqrt((1/46)* np.sum(np.square(LAI_Standard[:,200,300] - LAI_Inaccurate[:, 105,105]))) / 10
    # spa = np.sqrt((1/46)* np.sum(np.square(LAI_Standard[:, 105,105] - Spa_improvedLAI[:, 105,105]))) / 10
    # tem = np.sqrt((1/46)* np.sum(np.square(LAI_Standard[:, 105,105] - Tem_improvedLAI[:, 105,105]))) / 10
    # # improved = np.sqrt((1/46)* np.sum(np.square(LAI_Standard[:, 5, 5] / 10 - improved_5_5)))
    # print(inaccurate, spa, tem)
    # print(Spa_5_5)
    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        # 'title': 'pos_%s_%s' % (x_v, y_v),
        'title': '',
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [LAI_Standard[:, 350, 150]/10, LAI_Inaccurate[:, 350, 150]/10, Spa_5_5, Tem_5_5,],
        'le_name': ['Standard','Inaccurate', 'Temporal', 'Spatial',],
        'color': ['#958b8c',  '#bfdb39', '#ffe117', '#fd7400', '#7ba79c'],
        'marker': False,
        'size': False,
        'lineStyle': ['solid','solid','dashed','dashed','solid',],
        },'./Daily_cache/0530/Simu_Improved_(350, 150)', True, 2)

#计算一个点提升后的LAI值（之前的权重计算方法）
def previous_method():
    LAI_Standard = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    LAI_Inaccurate = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Inaccurate(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
    pos = [5,5]
    Tem_Lai = []
    Tem_Weight = []
    Spa_Lai = []
    Spa_Weight = []
    for index in range(1, 45):
        print(index)
        Tem = Improved_Pixel.Temporal_Cal(LAI_Inaccurate, index, pos, LandCover, Err_weight, 3, 8, 0.35)
        Spa = Improved_Pixel.Spatial_Cal(LAI_Inaccurate, index, pos, LandCover, Err_weight, 3, 8)
        Tem_Lai.append(Tem['filling'])
        Tem_Weight.append(Tem['weight'])
        Spa_Lai.append(Spa['filling'])
        Spa_Weight.append(Spa['filling'])
    
    Spa_5_5 = np.array(Spa_Lai) / 10
    Tem_5_5 =  np.array(Tem_Lai) / 10
    x = np.array(Spa_Weight)
    y = np.array(Tem_Weight)
    improved_5_5 = (Spa_5_5 * x + Tem_5_5 * y) / (x + y)

    inaccurate = np.sqrt((1/44)* np.sum(np.square(LAI_Standard[1:45, 5, 5] - LAI_Inaccurate[1:45, 5, 5]))) / 10
    spa = np.sqrt((1/44)* np.sum(np.square(LAI_Standard[1:45, 5, 5] / 10 - Spa_5_5))) 
    tem = np.sqrt((1/44)* np.sum(np.square(LAI_Standard[1:45, 5, 5] / 10 - Spa_5_5))) 
    improved = np.sqrt((1/44)* np.sum(np.square(LAI_Standard[1:45, 5, 5] / 10 - improved_5_5)))
    print(inaccurate, spa, tem, improved)

    Public_Methods.draw_polt_Line(np.arange(8, 353, 8),{
        # 'title': 'pos_%s_%s' % (x_v, y_v),
        'title': '',
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [LAI_Standard[1:45, 5, 5]/10, LAI_Inaccurate[1:45, 5, 5]/10, Spa_5_5, Tem_5_5, improved_5_5,],
        'le_name': ['Standard','Inaccurate', 'Temporal', 'Spatial', 'Improved',],
        'color': ['#958b8c', '#a4ac80', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c'],
        'marker': False,
        'size': False,
        'lineStyle': ['solid','solid','dashed','dashed','solid',],
        },'./Daily_cache/0506/Previous_Improved_(5,5)', True, 2)
    