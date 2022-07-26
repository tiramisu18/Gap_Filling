import numpy as np

# LAI数值转为0-7
def LAI_multiples():
    LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    for idx in range(0, 46):
        print(idx)
        for i in range(0, 500):
            for j in range(0,500):
                if LAI_Simu[idx][i][j] <= 70 : 
                    LAI_Simu[idx][i][j] = LAI_Simu[idx][i][j] / 10
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_noErr(0-7)', LAI_Simu)

# 将误差百分比转换为对应的权重
def set_err_weight():
    LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')   
    Err = np.load('./Simulation_Dataset/LAI/Simu_Method_3/Err_peren.npy')

    zero = np.zeros(46*500*500, dtype=int).reshape(46, 500, 500)
    bound = [0, 10, 20, 30, 40]
    weight = [10, 8, 6, 4, 2]
    for i in range(0,len(bound)):
        if i == 0: 
            pos = Err.__eq__(0)
            zero[pos] = 10
        else:
            pos = np.logical_and(Err > bound[i-1], Err <= bound[i])
            zero[pos] = weight[i]
    
    pos = LAI_Simu.__gt__(70)
    zero[pos] = 0
    np.save('./Simulation_Dataset/LAI/Simu_Method_3/Err_weight', zero)
