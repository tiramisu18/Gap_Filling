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

    # fig, ax = plt.subplots()
    # xdata, ydata = [], []
    # ln, = ax.plot([], [], 'r-', animated=True)
 
    # def init():
    #   ax.set_xlim(0, 2*np.pi)
    #   ax.set_ylim(-1, 1)
    #   return ln,
 
    # def update(frame):
    #   xdata.append(frame)
    #   ydata.append(np.sin(frame))
    #   ln.set_data(xdata, ydata)
    #   return ln,
    
    # ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
    # ani.save('sin_dot.gif', fps=30)
    # plt.show()

def draw_multiLine (data, key, url):
    aa = np.arange(1, 21, 1)
    # plt.xticks(aa)
    plt.figure(figsize=(15, 6)) #宽，高
    # plt.title('MRE', family='Times New Roman', fontsize=18)   
    plt.xlabel('WinSize', fontsize=15, family='Times New Roman') 
    plt.ylabel(key, fontsize=15, family='Times New Roman')
    line1=plt.plot(aa,data[0][key], label='count', color='#fd7400',  marker='o', markersize=5)
    line2=plt.plot(aa,data[1][key], label='count', color='#bfdb39',  marker='o', markersize=5)
    line3=plt.plot(aa,data[2][key], label='count', color='#016382',  marker='o', markersize=5)
    line4=plt.plot(aa,data[3][key], label='count', color='#1f8a6f',  marker='o', markersize=5)
    line5=plt.plot(aa,data[4][key], label='count',color='#ffe117',  marker='o', markersize=5)
    plt.legend(
    (line1[0],  line2[0],  line3[0],  line4[0],  line5[0]), 
    ('Pow_1', 'Pow_2','Pow_3','Pow_4','Pow_5',),
    loc = 1, prop={'size':15, 'family':'Times New Roman'},
    )
    plt.savefig(url, dpi=300)
    plt.show()

def draw_plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, '#fd7400')
    # plt.plot(x, y, 'r--', x, t**2, 'bs', t, t**3, 'g^')
    # plt.plot(t, y1, '#fd7400', t, y2, '#bfdb39')
    ax.set(xlabel='day', ylabel='LAI', title='title')
    # ax.grid()

    # fig.savefig("test.png")
    plt.show()    

def draw_plot_two(x, y1, y2, savePath, issave, title = 'title'):
    # plt.xticks(aa)
    # plt.figure(figsize=(15, 6)) #宽，高
    plt.title(title, family='Times New Roman', fontsize=18)   
    plt.xlabel('day', fontsize=15, family='Times New Roman') 
    plt.ylabel('LAI', fontsize=15, family='Times New Roman')
    line2=plt.plot(x,y1, label='count', color='#fd7400',  marker='o', markersize=3)
    line3=plt.plot(x,y2, label='count', color='#bfdb39',  marker='^', markersize=3)
    plt.legend(
    (line2[0],  line3[0]), 
    ('Original', 'Filling',),
    loc = 1, prop={'size':15, 'family':'Times New Roman'},
    )
    if issave :plt.savefig(savePath, dpi=300)
    plt.show()

def get_GreatPixel (MQC_Score, data):
    result_data = []
    for i in range(0, 2400):
        for j in range(0, 2400):
            if MQC_Score[i][j] > 80 and data[i][j] <= 70:
                result_data.append([i, j])
    return result_data

def random_pos ():
    # rand_i = int_random(0, 500, fill_pos_length)
    # rand_j = int_random(700, 1200, fill_pos_length)

    # print(aa)
    # print(bb)
    rand_i = [
    313, 307, 204, 370, 372, 414, 
    361, 433, 331, 10, 351, 403, 
    340, 197, 175, 377, 372, 66, 
    23, 337, 213, 21, 88, 449, 
    396, 147, 22, 336, 103, 236, 
    60, 23, 374, 156, 136, 412, 
    217, 90, 29, 246, 16, 261, 
    270, 264, 181, 376, 342, 27, 
    227, 342, 208, 247, 254, 168, 
    177, 469, 117, 83, 21, 209, 
    230, 399, 353, 302, 276, 373, 
    488, 380, 222, 404, 21, 231, 
    84, 358, 354, 286, 81, 15, 
    318, 262, 351, 184, 72, 474,
    454, 466, 482, 239, 241, 284, 
    487, 438, 195, 455, 390, 20, 
    347, 417, 496, 429]
    rand_j = [
    939, 940, 717, 1050, 839, 1085, 
    1080, 956, 935, 1169, 705, 1117, 
    936, 830, 1140, 942, 1160, 756, 
    752, 898, 1027, 706, 761, 1133, 
    778, 846, 1116, 881, 1043, 759, 
    800, 721, 944, 1001, 765, 1192, 
    1150, 1186, 727, 746, 715, 997, 
    942, 1042, 1005, 804, 1025, 1073, 
    1104, 1026, 1147, 796, 793, 833, 
    911, 718, 841, 878, 1033, 1035, 
    1071, 955, 1020, 830, 805, 1144, 
    975, 1082, 728, 1081, 754, 869, 
    1056, 1079, 1159, 861, 1155, 1108, 
    969, 945, 893, 956, 769, 1128, 
    959, 703, 883, 921, 1079, 704, 
    1149, 847, 1159, 984, 1012, 1170, 
    1156, 1083, 1157, 835]

    return {'i': rand_i, 'j': rand_j}

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

fileLists = ReadDirFiles.readDir(
  '../HDF/h11v04')
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

# fileIndex = 10
# render_MQC(MQC_All[fileIndex - 1])
# render_Img(fileDatas[fileIndex])

pixel_data = []
pixel_score = []
pixel_pos = {'x': 1200, 'y': 1200}
for i in range(2, 43):
    pixel_val = fileDatas[i][pixel_pos['x']][pixel_pos['y']] / 10
    pixel_mqc = MQC_All[i - 1][pixel_pos['x']][pixel_pos['y']]
    pixel_data.append(pixel_val)
    pixel_score.append(pixel_mqc)

# print(pixel_score)
print(pixel_data)
# draw_plot(np.arange(1, 45, 1), pixel_score)

# draw_plot(np.arange(2, 43, 1),pixel_data)
Filling_Pos = [[1200, 1200]]
Fil_val = []

# for index in range(2, 43):
#     re = Filling_Pixel.Fill_Pixel(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 3, 6)
#     Fil_val.append((re['Fil'])[0] / 10)
# print(Fil_val)


# draw_plot_two(np.arange(2, 43, 1),pixel_data, Fil_val, './Daily_cache/pos_1200_1200', True, 'pos_1200_1200')



for index in range(2, 43):
    re = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 3, 6)
    Fil_val.append((re['Fil'])[0] / 10)
print(Fil_val)


draw_plot_two(np.arange(2, 43, 1),pixel_data, Fil_val, './Daily_cache/pos_1200_1200_part_no_MQC', True, 'pos_1200_1200_part_no_MQC')