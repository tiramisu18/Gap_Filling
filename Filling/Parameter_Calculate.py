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
# from matplotlib import colors
import copy
import ReadDirFiles
import math
import h5py
import time
import random
import Filling_Pixel
import Public_Methods

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

def render_Img (data, title='Algo Path', issave=False, savepath=''):
    plt.imshow(data, cmap = plt.cm.coolwarm)  # cmap= plt.cm.jet
    # plt.imshow(data, cmap = plt.cm.coolwarm) 
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

# 求参数的最佳值_绘制折线图及空间图 （求46期的均值，若单独取某一期的数据会存在季节性变化的差异）
def get_better_para(simuStandLAI, fileDatas, landCover, qualityControl, type):
    # Spatial
    # standLAI = np.array(simuStandLAI).mean(axis=0)
    standLAI = np.array(simuStandLAI) 
    if type == 1: # Spatial
        HalfWidth = 9
        condition_1 = range(1, 6)
        condition_2 = range(1, HalfWidth)
    else: # Temporal
        HalfLength = 7
        SES_pow_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        condition_1 = SES_pow_array
        condition_2 = range(1, HalfLength)

    lineAll_RMSE = []
    lineAll_Std = []
    image_RMSE = []
    for c1 in condition_1:
        oneLine_RMSE = []
        oneLine_Std = []
        oneImage_RMSE = []
        for c2 in condition_2:
            print(c1,c2)
            oneYear = []
            for index in range(0, 46):
                result = Filling_Pixel.Spatial_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, c1, c2) if type == 1 else Filling_Pixel.Temporal_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, c2, c1)
                oneYear.append(result)
            # yearMean = np.array(oneYear).mean(axis=0)
            # calRMSE = np.sqrt((1/(len(fileDatas[0])*len(fileDatas[0][0]))) * np.sum(np.square(standLAI - yearMean)))
            yearMean = np.array(oneYear)
            TileRMSE = np.sqrt((1/46) * np.sum(np.square(standLAI - yearMean), axis=0))
            oneImage_RMSE.append(TileRMSE / 10)
            RMSE_Mean = np.mean(TileRMSE) / 10
            Std = np.std(TileRMSE, ddof=1) / 10
            oneLine_RMSE.append(RMSE_Mean)
            oneLine_Std.append(Std)
        lineAll_RMSE.append(oneLine_RMSE)
        lineAll_Std.append(oneLine_Std)
        image_RMSE.append(oneImage_RMSE)

    return {'RMSE': lineAll_RMSE, 'Std':lineAll_Std, 'RMSE_Image': image_RMSE}


# 将参数绘制成折线图
def parameter_line(paraType, lineAll):
    if paraType == 1: # Spatial
        x = np.arange(1, 9, 1)
        xlable = 'Half Width'
        le_name = ['cc=1', 'cc=2', 'cc=3', 'cc=4', 'cc=5']
        savepath = './Daily_cache/0518/Spatial_Para_(100,300)_err'
    else : #Temporal
        x = np.arange(1, 7, 1)
        xlable = 'Half Length'
        le_name = ['cc=0.2', 'cc=0.3','cc=0.4', 'cc=0.5', 'cc=0.6', 'cc=0.7']
        savepath = './Daily_cache/0518/Temporal_Para_(100,300)_err'

    color_arr = ['#548bb7', '#958b8c', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c', '#016382', '#dd8146', '#a4ac80', '#d9b15c', '#1f8a6f', '#987b2d']
    marker_arr = ['s', 'o', '.', '^', ',', 'v', '8', '*', 'H', '+', 'x', '_']
    plt.figure(figsize=(8,4))
        
    plt.title('', family='Times New Roman', fontsize=18)   
    plt.xlabel(xlable, fontsize=15, family='Times New Roman') 
    plt.ylabel('RMSE', fontsize=15, family='Times New Roman')
    line_arr = []
    for i in range(0, len(lineAll['RMSE'])):
        # line_arr.append((plt.plot(x,lineAll['RMSE'][i], label='count', color=color_arr[i],  marker=marker_arr[i], markersize=3))[0])
        line_arr.append((plt.errorbar(x, lineAll['RMSE'][i], yerr=lineAll['Std'][i], label='count', color=color_arr[i], linewidth=1, linestyle='dotted', marker=marker_arr[i], markersize=3))[0])
        # plt.errorbar(np.arange(1, 7, 1), lineAll['RMSE'][0], yerr=lineAll['Std'][0], label='both limits (default)')
        # ax.fill_between(np.arange(1, 7, 1), lineAll['Std'][i][0], lineAll['Std'][i][1] , alpha=.3, linewidth=0, color=color_arr[i])       
    plt.legend(
        (line_arr), 
        (le_name),
        loc = 1, prop={'size':15, 'family':'Times New Roman'},
        )
    plt.savefig(savepath, dpi=300)       
    plt.show()

# 将参数绘制成空间图
def parameter_images(paraType, lineAll):
    # plt.figure(figsize=(10,4))
    if paraType == 1:
        con = (5, 8)
        figsize=(7,4)
        savepath = './Daily_cache/0522/Spatial_Para_(100,300)_combine_3'
    else:
        con = (6, 6)
        figsize=(9,8)
        savepath = './Daily_cache/0522/Temporal_Para_(100,300)_combine'

    fig, axs = plt.subplots(con[0], con[1], figsize=figsize)
    # fig.suptitle('Multiple images')
    images = []
    for i in range(con[0]):
        for j in range(con[1]):
            # Generate data with a range that varies from one plot to the next.
            # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
            images.append(axs[i, j].imshow(lineAll['RMSE_Image'][i][j], cmap = plt.cm.hsv))
            axs[i, j].axis('off')
            # axs[i, j].width = 

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs,  fraction=.1)
    plt.savefig(savepath, dpi=300)
    plt.show()


simuStandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
fileDatas = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
Err_weight= np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
left = 100
right = 300
paraType = 2
# lineAll = get_better_para(simuStandLAI[:,left:right,left:right], fileDatas[:,left:right,left:right], LandCover[left:right,left:right], Err_weight[:,left:right,left:right], paraType)

# np.save('./Daily_cache/0522/Temporal_(100,300)', lineAll)
# Spatial = np.load('./Daily_cache/0522/Spatial_(100,300).npy', allow_pickle=True).item()
lineAll = np.load('./Daily_cache/0522/Temporal_(100,300).npy', allow_pickle=True).item()
parameter_line(paraType, lineAll)
# parameter_images(paraType, lineAll)
