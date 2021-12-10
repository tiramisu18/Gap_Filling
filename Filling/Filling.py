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

def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    return LAI

# 生成一个包含n个介于a和b之间随机整数的数组
def int_random(a, b, n) :
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        if(d_int not in a_list) :
            a_list.append(d_int)
        else :
            pass
    return a_list


def render_MQC (MQC_Score, title='MQC'):
    plt.imshow(MQC_Score, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    # plt.savefig('test.png', dpi=300)
    plt.show()

def render_Img (data, title='Image'):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title(title, family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
    plt.show()

def render_GIF (data, title='Image'):
    fig, ax = plt.subplots()
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    ax.title(title, family='Times New Roman', fontsize=18)
    ax.imshow(data, cmap=cmap, norm=norm)
    cbar = ax.colorbar()
    cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
    plt.show()   

def get_GreatPixel (MQC_Score, data):
    result_data = []
    for i in range(0, 2400):
        for j in range(0, 2400):
            if MQC_Score[i][j] > 80 and data[i][j] <= 70:
                result_data.append([i, j])
    return result_data

def read_MQC (path, savepath):
    MQC_File=h5py.File(path) 
    # print(MQC_File.keys())
    file_Content = MQC_File["MQC_Score"]

    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    MQC_All = []
    for idx in range(0, 44):
        MQC_data = MQC_File[file_Content[0,idx]]  # [column, row]
        MQC_Score = np.transpose(MQC_data[:])
        MQC_All.append(MQC_Score)
    print(len(MQC_All))
    print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    np.save(savepath, MQC_All)

fileLists = ReadDirFiles.readDir('../HDF/h11v04')
# print('lists', len(fileLists))
fileDatas = []
for file in fileLists:
  fileDatas.append(ReadFile(file))


LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h11v04.006.2019199203448.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
LC_info = gdal.Open(LC_subdatasets[0][0]).ReadAsArray()



# read MQC file
# read_MQC('../MQC/h11v04_2018_MQC_Score_part.mat', './MQC_NP/h11v04_2018_part')          

# MQC_All = np.load('./MQC_NP/h11v04_2018.npy')
MQC_All = np.load('./MQC_NP/h11v04_2018_part.npy')

fileIndex = 10
# render_MQC(MQC_All[fileIndex - 1])
# render_Img(fileDatas[fileIndex])
x_v = 1200
y_v = 1200
pixel_data = []
pixel_score = []
pixel_pos = {'x': x_v, 'y': y_v}
for i in range(2, 43):
    pixel_val = fileDatas[i][pixel_pos['x']][pixel_pos['y']] / 10
    # pixel_mqc = MQC_All[i - 1][pixel_pos['x']][pixel_pos['y']]
    pixel_data.append(pixel_val)
    # pixel_score.append(pixel_mqc)

# print(pixel_score)
# print(pixel_data)
# draw_plot(np.arange(1, 45, 1), pixel_score)

# draw_plot(np.arange(2, 43, 1),pixel_data)
Filling_Pos = [[x_v, y_v]]
Fil_val_1 = []
Fil_val_2 = []
Fil_val_3 = []
Tem_W = []
Spa_W = []
Mqc_W = []


for index in range(2, 43):
    # re1 = Filling_Pixel.Fill_Pixel(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 2, 6)
    re1 = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 2, 6, 2) # no MQC
    Fil_val_1.append(re1['Tem'][0] / 10)
    Fil_val_2.append(re1['Spa'][0] / 10)
    Fil_val_3.append(re1['Fil'][0] / 10)
    Tem_W.append(re1['T_W'][0])
    Spa_W.append(re1['S_W'][0])
    Mqc_W.append(re1['M_W'][0])
# print(Fil_val)
# Draw_PoltLine.draw_Line(np.arange(2, 43, 1),pixel_data, Fil_val_1, Fil_val_2, Fil_val_3, './Daily_cache/pos_%s_%s_indep_part' % (x_v, y_v), False, 'pos_%s_%s_indep_part' % (x_v, y_v))

Draw_PoltLine.draw_polt_Line(np.arange(2, 43, 1),{
    'title': 'pos_%s_%s_W' % (x_v, y_v),
    'xlable': 'Day',
    'ylable': 'Weight',
    'line': [Tem_W, Spa_W, Mqc_W],
    'le_name': ['Tem', 'Spa', 'Mqc'],
    'color': False,
    'marker': False,
    'lineStyle': []
    },'./Daily_cache/pos_%s_%s' % (x_v, y_v), False, 2)

Draw_PoltLine.draw_polt_Line(np.arange(2, 43, 1),{
    'title': 'pos_%s_%s_indep_part' % (x_v, y_v),
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': [pixel_data, Fil_val_1, Fil_val_2, Fil_val_3],
    'le_name': ['Original', 'Tem', 'Spa', 'Fil'],
    'color': ['gray', '#bfdb39', '#ffe117', '#fd7400'],
    'marker': False,
    'lineStyle': ['dashed']
    },'./Daily_cache/pos_%s_%s_indep_part' % (x_v, y_v), False, 2)
# MQC_All = np.load('./MQC_NP/h11v04_2018_part.npy')
# for index in range(2, 43):
#     re2 = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 2, 6, 2) 
#     re3 = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 2, 6, 1) # no MQC
#     Fil_val_2.append((re2['Fil'])[0] / 10)
#     Fil_val_3.append((re3['Fil'])[0] / 10)

# draw_multiLine(np.arange(2, 43, 1),pixel_data, Fil_val_1, Fil_val_2, Fil_val_3, './Daily_cache/pos_%s_%s_all_1205' % (x_v, y_v), False, 'pos_%s_%s_all' % (x_v, y_v))


# Fil_val = []
# for index in range(2, 43):
#     re = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 3, 12, 0.35, 3, 6 , 1)
#     Fil_val.append((re['Fil'])[0] / 10)
# print(Fil_val)
# draw_plot_two(np.arange(2, 43, 1),pixel_data, Fil_val, './Daily_cache/pos_%s_%s_part_no_MQC' % (x_v, y_v), False, 'pos_%s_%s_part_no_MQC' % (x_v, y_v))