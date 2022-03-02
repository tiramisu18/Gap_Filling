import os
from tokenize import String
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
from scipy import stats, signal
from scipy.interpolate import interp1d
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
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC}

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


def render_Img (QC_data, title='', issave=False, savepath=''):
    plt.imshow(QC_data, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    # plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

def render_LAI (data, title='Image', issave=False, savepath=''):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title(title, family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','1','2','3','4','5','6','7','250'])
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()
  

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

def read_QC(QC):
    QC_Bin = []
    for idx in range(0, 46):
        print(idx)
        file_bin = []
        for i in range(0, 2400):
            weight = 0 
            row_bin = []
            for j in range(0, 2400):
                one = '%08d' % int((bin(round(float((QC[idx][i][j])))).split('b')[1]))
                row_bin.append(one)
            file_bin.append(row_bin)
        QC_Bin.append(file_bin)
    np.save('../QC/h11v04_2018_Bin', QC_Bin)

def QC_AgloPath(QCBin):
    QC_Wei = []
    for idx in range(0, 46):
        print(idx)
        file_wei = []
        for i in range(0, 2400):
            weight = 0 
            row_wei = []
            for j in range(0, 2400):
                one = str(QCBin[idx][i][j])[0:3]
                if one == '000' or  one == '001': weight = 10
                elif one == '010' or one == '011' : weight = 5
                else: weight = 0
                row_wei.append(weight)
            file_wei.append(row_wei)
        QC_Wei.append(file_wei)
    np.save('../QC/h11v04_2018_AgloPath_Wei', QC_Wei)

def QC_CloudState(QCBin):
    QC_Wei = []
    for idx in range(0, 46):
        print(idx)
        file_wei = []
        for i in range(0, 2400):
            weight = 0 
            row_wei = []
            for j in range(0, 2400):
                one = str(QCBin[idx][i][j])[3:5]
                if one == '00' : weight = 10
                elif one == '01' : weight = 6
                elif one == '10' : weight = 3
                else: weight = 0
                row_wei.append(weight)
            file_wei.append(row_wei)
        QC_Wei.append(file_wei)
    np.save('../QC/h11v04_2018_CloudState_Wei', QC_Wei)


def get_GreatPixel (QC, data, len):
    result_data = []
    for i in range(0, 2400):
        for j in range(0, 2400):
            if QC[i][j] == 10 and data[i][j] <= 70:
                result_data.append([i, j])
                if len(result_data) > len: return result_data
    # return result_data

def random_pos(QC, ran_len, length):
    rand_pos_1 = int_random(0, 2399, ran_len)
    rand_pos_2 = int_random(0, 2399, ran_len)
    fill_pos = []
    for ele in range(0, ran_len):
        if QC[rand_pos_1[ele]][rand_pos_2[ele]] == 10:
            fill_pos.append([rand_pos_1[ele], rand_pos_2[ele]])
            if len(fill_pos) == length: return fill_pos

def get_mode_mean(dataList):
    result_list = []
    i_count = 0
    while len(dataList) > 0:
        i_count += 1
        num = stats.mode(dataList)[0][0]
        result_list.append(num)
        num_count = str(dataList).count('%s' % num)
        for i in range(0, int(num_count)):
            dataList.remove(num)
        if i_count == 3: dataList = []
    # return {'result_list': result_list, 'ori': dataList }   
    return round(np.mean(result_list), 2) 

def get_vege_linedata(vege_type, fileDatas, LC_info, QC_All):
    LC_part = []
    index_arr_num = []
    for i in range(500, 1000):
        row = []
        for j in range(1000, 1500):
            oneof = LC_info[i][j]
            if oneof == vege_type: index_arr_num.append([i,j])
            row.append(oneof)
        LC_part.append(row)

    # print(str(LC_info).count('5'))
    # print(index_arr_num, len(index_arr_num))
    # render_Img(LC_part)

    vegeType_LAI = []
    for day_idx in range(0, 46):
        tile_one = []
        for ele in index_arr_num:
            pos_one = fileDatas[day_idx][ele[0]][ele[1]]
            if pos_one <= 70 and QC_All[day_idx][ele[0]][ele[1]] == 10: tile_one.append(pos_one/10)
            # if pos_one <= 70: tile_one.append(pos_one/10)
        vegeType_LAI.append(tile_one)  

    # print(vegeType_LAI, len(vegeType_LAI[0]))  
    # print(round(np.mean(vegeType_LAI[0]), 2))

    vege_line = []
    period_len = []
    for period in range(0, 46):
        # print(period)
        if len(vegeType_LAI[period]) > 0:
            vege_line.append(round(np.mean(vegeType_LAI[period]), 2))
            # vege_line.append(stats.mode(vegeType_LAI[period])[0][0])
            # vege_line.append(get_mode_mean(vegeType_LAI[period]))
        else:
            vege_line.append(0)
        period_len.append(len(vegeType_LAI[period]))
    return vege_line

fileLists = ReadDirFiles.readDir('../HDF/h11v04')
# print('lists', len(fileLists))

fileDatas = []
QCDatas = []
for file in fileLists:
    result = ReadFile(file)
    fileDatas.append(result['LAI'])
    QCDatas.append(result['QC'])


LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h11v04.006.2019199203448.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
LC_info = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()


fileIndex = 7

QC_All = np.load('../QC/h11v04_2018_AgloPath_Wei.npy')


vege_type_list = [1,3,4,6,7,8]
all_vege_data = []
for i in vege_type_list:
    line = get_vege_linedata(i, fileDatas, LC_info, QC_All)
    vege_line = signal.savgol_filter(line, 10, 2) 
    all_vege_data.append((vege_line))

Draw_PoltLine.draw_polt_Line(np.arange(1, 47, 1),{
    'title': 'Vege_Line',
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': all_vege_data,
    'le_name': ['B1', 'B3', 'B4', 'B6', 'B7', 'B8'],
    'color': False,
    'marker': False,
    'lineStyle': []
    },'./Daily_cache/0229/vegeType_All', True, 2)
# vege_type = 1

# vege_line = get_vege_linedata(vege_type, fileDatas, LC_info, QC_All)
# vege_line = signal.savgol_filter(vege_line, 10, 2) 
# 99是滤波器窗口的长度（即系数的数目）。窗口长度必须是正奇数整数。一般是修改这个系数，如果你想平滑程度高一点，就提高这个系数；平滑程度低一点就降低这个系数。但是数值一定只能是奇数。
# # 1是拟合样本的多项式的阶数

# def smooth_curve(points, factor=0.5):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return  smoothed_points

# vege_line = smooth_curve(vege_line)


# 统计各植被类型像元个数
# print(str(LC_part).count("1"))
# print(str(LC_part).count("2"))
# print(str(LC_part).count("3"))
# print(str(LC_part).count("4"))
# print(str(LC_part).count("5"))
# print(str(LC_part).count("6"))
# print(str(LC_part).count("7"))
# print(str(LC_part).count("8"))




# Draw_PoltLine.draw_polt_Line(np.arange(1, 47, 1),{
#     'title': 'B%s'% vege_type,
#     'xlable': 'Day',
#     'ylable': 'LAI',
#     'line': [vege_line],
#     'le_name': ['Original', 'Tem', 'Spa', 'Fil'],
#     'color': ['gray', '#bfdb39', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
#     'marker': False,
#     'lineStyle': ['dashed']
#     },'./Daily_cache/0229/vegeType_B%s_mean_filter' % vege_type, True, 2)

