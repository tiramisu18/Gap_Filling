import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import Public_Methods
   

def tile_mean(tiles):
    # print(np.mean(tiles[idx].flatten()))
    count = []
    for day in range(0, 46):
        print(day)
        vals = 0
        vals_co = 0
        for i in range(0,500):
            for j in range(0, 500):
                val = tiles[day][i][j]
                if val <= 70 : 
                    vals += tiles[day][i][j]
                    vals_co += 1
        count.append(round((vals/vals_co) / 10,1))
    
    return count

def vege_type_mean(datas, lc):
    B1_all = []
    B3_all = []
    B4_all = []
    B6_all = []
    B7_all = []
    B8_all = []
    for day in range(0,46):
        print(day)
        type_B1 = []
        type_B3 = []
        type_B4 = []
        type_B6 = []
        type_B7 = []
        type_B8 = []
        for i in range(0, 500):
            for j in range(0, 500):
                oneof = lc[i][j]
                if oneof == 1: type_B1.append(datas[day][i][j])
                if oneof == 3: type_B3.append(datas[day][i][j])
                if oneof == 4: type_B4.append(datas[day][i][j])
                if oneof == 6: type_B6.append(datas[day][i][j])
                if oneof == 7: type_B7.append(datas[day][i][j])
                if oneof == 8: type_B8.append(datas[day][i][j])
        B1_all.append(round(np.mean(type_B1) / 10, 2))
        B3_all.append(round(np.mean(type_B3) / 10, 2))
        B4_all.append(round(np.mean(type_B4) / 10, 2))
        B6_all.append(round(np.mean(type_B6) / 10, 2))
        B7_all.append(round(np.mean(type_B7) / 10, 2))
        B8_all.append(round(np.mean(type_B8) / 10, 2))
    
    return {'B1': B1_all, 'B3': B3_all, 'B4': B4_all, 'B6': B6_all, 'B7': B7_all,'B8': B8_all}
    

LAI_ori = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
# LAI_Simu = np.load('../Simulation/Simulation_Dataset/LAI_Simu_noErr_10.npy')
LAI_Simu_2 = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
# LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')
# ori_mean = tile_mean(LAI_ori)
# simu_mean = tile_mean(LAI_Simu)
# simu_mean_2 = tile_mean(LAI_Simu_2)
# simu_addErr_mean = tile_mean(LAI_Simu_addErr)
LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
ori = vege_type_mean(LAI_ori, LandCover)
simu = vege_type_mean(LAI_Simu_2, LandCover)


vege_type_list = [1,3,4,6,7,8]
for type in vege_type_list:
    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        'title': 'B%s'% type,
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [ori['B%s' % type], simu['B%s' % type]],
        'le_name': ['Original', 'Improved', 'Simu2', 'Step2', 'B7', 'B8'],
        'color': [ '#bfdb39','#fd7400', '#ffe117', '#958b8c','#7ba79c'],
        'marker': ['o', '^', ',', '.' ],
        'lineStyle': ['dashed', 'dashed'],
        'size': {'width': 6, 'height': 4}
        },'./Daily_cache/0320/vege_lai_mean_B%s' % type, True, 2)