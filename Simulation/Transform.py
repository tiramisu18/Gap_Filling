import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import random


# LAI数值转为0-7
def LAI_multiples():
    LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    for idx in range(0, 46):
        print(idx)
        for i in range(0, 500):
            for j in range(0,500):
                if LAI_Simu[idx][i][j] <= 70 : 
                    LAI_Simu[idx][i][j] = LAI_Simu[idx][i][j] / 10
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_noErr(0-7)', LAI_Simu)

# 将误差百分比转换为对应的权重
def set_err_weight():
    LAI_Simu = np.load('./Simulation_Dataset/LAI_Simu_noErr.npy') # 不修改    
    Err = np.load('./Simulation_Dataset/LAI/Simu_Method_2/Err_peren.npy')
    for day in range(0,46):
        print(day)
        for i in range(0, 500):
            for j in range(0,500):
                if LAI_Simu[day][i][j] <= 7:
                    val = Err[day][i][j]
                    if val == 0 : Err[day][i][j] = 10
                    elif val > 0 and val <= 50 : Err[day][i][j] = 8
                    elif val > 50 and val <= 150 : Err[day][i][j] = 6
                    elif val > 150 and val <= 300 : Err[day][i][j] = 4
                    elif val > 300 and val <= 500 : Err[day][i][j] = 2
                    else: Err[day][i][j] = 0
                else: Err[day][i][j] = 0
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/Err_weight', Err)

