import os
import numpy as np
import numpy.ma as ma
import ReadDirFiles
import math
import Improved_Pixel
import Public_Methods
import Public_Methods
import time
from pathlib import Path

# 模拟数据的时空提升
url = '../Improved/Improved_SimuData'

LAI_Standard = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
LAI_Inaccurate = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
qualityControl= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')

dirList = ['Temporal', 'Spatial', 'Temporal_Weight', 'Spatial_Weight', 'Improved']       
if not Path(url).is_dir(): 
    os.mkdir(url)
    for ele in dirList:
        os.mkdir(f'{url}/{ele}')
else:
    for ele in dirList:
        if not Path(f'{url}/{ele}').is_dir(): os.mkdir(f'{url}/{ele}')

tem_list, spa_list = [], []
for index in range(46): 
    print(index)
    Tem = Improved_Pixel.Temporal_Cal(LAI_Inaccurate, index, LandCover, qualityControl, 3,  0.5)
    np.save(f'{url}/Temporal/LAI_{index + 1}', Tem)
    Spa = Improved_Pixel.Spatial_Cal(LAI_Inaccurate, index, LandCover, qualityControl, 2,  4)
    np.save(f'{url}/Spatial/LAI_{index + 1}', Spa)
    tem_list.append(Tem)
    spa_list.append(Spa)
temLAI = np.array(tem_list)
spaLAI = np.array(spa_list)

for index in range(46):
    if index == 0 or index == 45:
        one = (ma.masked_greater(temLAI[index], 70) + ma.masked_greater(spaLAI[index], 70)) / 2
        pos = LAI_Standard[index].__gt__(70)
        one[pos] = LAI_Standard[index][pos]
        np.save(f'{url}/Improved/LAI_{index + 1}', np.array(one))
    else:
        temWeight = Improved_Pixel.cal_TSS(temLAI, index)
        spaWeight = Improved_Pixel.cal_TSS(spaLAI, index)
        np.save(f'{url}/Spatial_Weight/LAI_{index + 1}', spaWeight)            
        np.save(f'{url}/Temporal_Weight/LAI_{index + 1}', temWeight)
        one = (ma.masked_greater(temLAI[index], 70) * temWeight + ma.masked_greater(spaLAI[index], 70) * spaWeight) / (temWeight + spaWeight)
        pos = LAI_Standard[index].__gt__(70)
        one[pos] = LAI_Standard[index][pos]
        np.save(f'{url}/Improved/LAI_{index + 1}', np.array(one))
        
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
