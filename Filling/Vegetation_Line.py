from copy import deepcopy
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import matplotlib.patches as patches
from matplotlib import animation 
from scipy import stats, signal, interpolate
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

    # render_Img(LC_part)
    # 统计各植被类型像元个数
    # print(str(LC_part).count("1"))
    # print(str(LC_part).count("2"))
    # print(str(LC_part).count("3"))
    # print(str(LC_part).count("4"))
    # print(str(LC_part).count("5"))
    # print(str(LC_part).count("6"))
    # print(str(LC_part).count("7"))
    # print(str(LC_part).count("8"))

    # 收集每一期所有算法路径为主算法的LAI值
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

# 简单平滑
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return  smoothed_points

# fileLists = ReadDirFiles.readDir('../HDF/h11v04')
# # print('lists', len(fileLists))

# fileDatas = []
# QCDatas = []
# for file in fileLists:
#     result = ReadFile(file)
#     fileDatas.append(result['LAI'])
#     QCDatas.append(result['QC'])


# LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h11v04.006.2019199203448.hdf')
# LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
# LC_info = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()


# QC_All = np.load('../QC/h11v04_2018_AgloPath_Wei.npy')


# 独立原始LAI数据
# LAI_data = []
# for  day in range(0, 46):
#     print(day)
#     day_arr = []
#     for i in range(500, 1000):
#         row = []
#         for j in range(1000, 1500):
#             oneof = fileDatas[day][i][j]
#             row.append(oneof)
#         day_arr.append(row)
#     LAI_data.append(day_arr)

# np.save('../Simulation/Simulation_Dataset/LAI_Ori', LAI_data)

# Vage_All = np.load('../Simulation/Simulation_Dataset/Vege_data.npz', allow_pickle=True)
# LC_part = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
# LAI_Ori = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')


# 生成模拟数据
# LAI_data = []
# for  day in range(0, 46):
#     print('time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print(day)
#     day_arr = []
#     for i in range(0, 500):
#         row = []
#         for j in range(0, 500):
#             LC_v = LC_part[i][j]
#             lai_one = LAI_Ori[day][i][j]
#             if lai_one < 70 : lai_one = lai_one/10
#             if LC_v == 1 : lai_one = round(Vage_All['B1'][day],1)
#             elif LC_v == 3 : lai_one = round(Vage_All['B3'][day],1)
#             elif LC_v == 4 : lai_one = round(Vage_All['B4'][day],1)
#             elif LC_v == 6 : lai_one = round(Vage_All['B6'][day],1)
#             elif LC_v == 7 : lai_one = round(Vage_All['B7'][day],1)
#             elif LC_v == 8 : lai_one = round(Vage_All['B8'][day],1)
#             print(lai_one)
#             # if LC_v == 1 or LC_v == 3 or LC_v == 4 or LC_v == 6 or LC_v == 7 or LC_v == 8: lai_one = round(Vage_All['B%s'%LC_v][day],1)
#             row.append(lai_one)
#         day_arr.append(row)
#     LAI_data.append(day_arr)

# np.save('../Simulation/Simulation_Dataset/LAI_Simu_noErr', LAI_data)


# # 得到多个植被年际变化 绘制所有植被类型 滤波平滑处理
# vege_type_list = [1,3,4,6,7,8]
# all_vege_data = []
# for i in vege_type_list:
#     line = get_vege_linedata(i, fileDatas, LC_info, QC_All)
#     vege_line = signal.savgol_filter(line, 25, 6, mode='mirror') 
#     # vege_line = signal.savgol_filter(line, 11, 2, mode='nearest') 
#     all_vege_data.append((vege_line))

# print(Vege_data)
# np.savez('../Simulation/Simulation_Dataset/Vege_data', 
#     B1=all_vege_data[0], B3=all_vege_data[1], B4=all_vege_data[2], B6=all_vege_data[3], B7=all_vege_data[4], B8=all_vege_data[5])


# Draw_PoltLine.draw_polt_Line(np.arange(1, 47, 1),{
#     'title': 'Vege_Line',
#     'xlable': 'Day',
#     'ylable': 'LAI',
#     'line': all_vege_data,
#     'le_name': ['B1', 'B3', 'B4', 'B6', 'B7', 'B8'],
#     'color': False,
#     'marker': False,
#     'lineStyle': []
#     },'./Daily_cache/0309/vegeType_All_26_6_mirror', True, 2)

# 得到单个植被年际变化、滤波平滑处理、绘制单个植被曲线
# vege_type = 8
# vege_line = get_vege_linedata(vege_type, fileDatas, LC_info, QC_All)
# # print(vege_line)
# vege_line = signal.savgol_filter(vege_line, 25, 6, mode='mirror') # 25是滤波器窗口的长度（即系数的数目）# 6是拟合样本的多项式的阶数
# # vege_line = signal.savgol_filter(vege_line, 11, 2) 
# print(vege_line)

# Draw_PoltLine.draw_polt_Line(np.arange(1, 47, 1),{
#     'title': 'B%s'% vege_type,
#     'xlable': 'Day',
#     'ylable': 'LAI',
#     'line': [vege_line],
#     'le_name': ['Original', 'Tem', 'Spa', 'Fil'],
#     'color': ['gray', '#bfdb39', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
#     'marker': False,
#     'lineStyle': ['dashed']
#     },'./Daily_cache/0309/vegeType_B%s_mirror' % vege_type,True, 2)

# 插值方法
# line = get_vege_linedata(vege_type, fileDatas, LC_info, QC_All)
# x = np.arange(46)
# y = line
# cs = interpolate.CubicSpline(x, y)
# xs = np.arange(1,46, 1)
# fig, ax = plt.subplots()
# ax.plot(x, y, 'o', label='data')
# # ax.plot(xs, np.sin(xs), label='true')
# ax.plot(xs, cs(xs), label="S")
# ax.legend(loc='lower left', ncol=2)
# plt.show()


# 模拟数据集方法2
# 保留所有主算法像元数据，其余用标准曲线值代替，对所有像元单独使用滤波算法平滑
def get_simu():
    Vage_All = np.load('../Simulation/Simulation_Dataset/Vege_data.npz', allow_pickle=True)
    LC_part = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    LAI_Ori = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
    QC_All = np.load('../QC/h11v04_2018_AgloPath_Wei.npy')

    LAI_new = []
    for day in range(0, 46):
        print('time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(day)
        day_arr = []
        for i in range(0, 500):
            row = []
            for j in range(0, 500):
                lai_one = LAI_Ori[day][i][j]
                lc = LC_part[i][j]
                if lai_one < 70 and QC_All[day][i+500][j+1000] < 10: 
                    if lc == 1 : lai_one = round(Vage_All['B1'][day],1) * 10
                    elif lc == 3 : lai_one = round(Vage_All['B3'][day],1) * 10
                    elif lc == 4 : lai_one = round(Vage_All['B4'][day],1) * 10
                    elif lc == 6 : lai_one = round(Vage_All['B6'][day],1) * 10
                    elif lc == 7 : lai_one = round(Vage_All['B7'][day],1) * 10
                    elif lc == 8 : lai_one = round(Vage_All['B8'][day],1) * 10
                row.append(lai_one)
            day_arr.append(row)
        LAI_new.append(day_arr)
    np.save('../Simulation/Simulation_Dataset/Test/LAI_Simu_Step1', LAI_new)

def pixel_SG():
    LC_part = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    Simu_Step1 = np.load('../Simulation/Simulation_Dataset/Method_2/LAI_Simu_Step1.npy')
    Simu_Step2 = deepcopy(Simu_Step1)
    for i in range(0, 500):
        print(i)
        for j in range(0, 500):
            lc = LC_part[i][j]
            if lc == 1 or lc == 3 or lc == 4 or lc == 6 or lc == 7 or lc == 8:
                line = []
                for idx in range(0, 46):
                    line.append(Simu_Step1[idx][i][j])
                # print(line)
                sg_line = signal.savgol_filter(line, 25, 6, mode='mirror')
                # print(sg_line[1])
                for order in range(0, 46):
                    if sg_line[order] < 0:
                        Simu_Step2[order][i][j] = 0
                    else:
                        Simu_Step2[order][i][j] = round(sg_line[order],0)
                    # print(Simu_Step2[order][i][j])

    np.save('../Simulation/Simulation_Dataset/Method_2/LAI_Simu_Step2', Simu_Step2)
   

LAI_Ori = np.load('../Simulation/Simulation_Dataset/LAI_Ori.npy')
LAI_Simu = np.load('../Simulation/Simulation_Dataset/LAI_Simu_noErr_10.npy')

Simu_Step1 = np.load('../Simulation/Simulation_Dataset/Method_2/LAI_Simu_Step1.npy')
Simu_Step2 = np.load('../Simulation/Simulation_Dataset/Method_2/LAI_Simu_Step2.npy')
# QC_All = np.load('../QC/h11v04_2018_AgloPath_Wei.npy')
# print(Simu_Step1[23][10][10])
# print(Simu_Step2[23][10][10])
aa = []
bb = []
cc = []
dd = []
x_v = 0
y_v = 0
# (0,3) (2,1) （2，2） (499, 499)
for i in range(0, 46):   
    aa.append(LAI_Ori[i][x_v][y_v] )
    bb.append(LAI_Simu[i][x_v][y_v] )
    cc.append(Simu_Step1[i][x_v][y_v])
    dd.append(Simu_Step2[i][x_v][y_v] )
    # dd.append(QC_All[i][500+x_v][1000+y_v])
# print(aa)
# print(bb)
# print(cc)
# print(dd)
Draw_PoltLine.draw_polt_Line(np.arange(1, 47, 1),{
    'title': 'Vege_Line',
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': [aa, bb, cc, dd],
    'le_name': ['Ori', 'Mean', 'Step1', 'Step2', 'B7', 'B8'],
    'color': ['#bfdb39', '#958b8c', '#fd7400', '#ffe117','#7ba79c'],
    'marker': [',', ',', '^', '.' ],
    'lineStyle': ['dashed', 'dashed', '']
    },'./Daily_cache/0315/ori_simu_step1', False, 2)