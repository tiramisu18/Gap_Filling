# 探究空间相关性中的参数最佳值
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
import copy
import ReadDirFiles
import math
import h5py
import time
import random

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

def Spatial_W2 (fileDatas, index, Filling_Pos, LC_Value, MQC_File, euc_pow, fill_pos_length):
    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # aa = int_random(0, 649146, 100)
    # print(aa)
    test_posi_nan=[]
    test_posi_nan = Filling_Pos   
    # test_posi_nan = Filling_Pos[1000600: 1000700]
    # print(len(test_posi_nan), test_posi_nan)
    spa_result_inter = 0
    spa_weight = 0
    spa_cu_dataset = fileDatas[index]
    spa_cu_before_dataset = fileDatas[index - 1]
    spa_cu_after_dataset = fileDatas[index + 1]

    # spa_winSize_unilateral = 10 # n*2 + 1

    MQC_Score = MQC_File[index - 1]
    MQC_Score_before = MQC_File[index - 2]
    MQC_Score_after = MQC_File[index]

    w_array = []
    F_value = []
    O_value = []
    # euc_pow = 6
    # 反距离加权（欧氏距离）的窗口大小
    for winsize in range(1, 21):
        spa_winSize_unilateral = winsize
        oneSizeWeight_arr = []
        one_F_value = []
        one_O_value = []
        for pos in test_posi_nan:    
            # print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            lc_type = LC_Value[pos[0]][pos[1]]    
            or_before_value = spa_cu_before_dataset[pos[0]][pos[1]]
            or_after_value = spa_cu_after_dataset[pos[0]][pos[1]]
            or_value = spa_cu_dataset[pos[0]][pos[1]]      
            spa_row_before = 0
            spa_row_after = 2400
            spa_col_before = 0
            spa_col_after = 2400
            if pos[0] - spa_winSize_unilateral > 0 : spa_row_before = pos[0] - spa_winSize_unilateral
            if pos[0] + spa_winSize_unilateral < 2400 : spa_row_after = pos[0] + spa_winSize_unilateral
            if pos[1] - spa_winSize_unilateral > 0 : spa_col_before = pos[1] - spa_winSize_unilateral
            if pos[1] + spa_winSize_unilateral < 2400 : spa_col_after = pos[1] + spa_winSize_unilateral
        
            numerator = [0] * 3 # 分子
            denominator = [0] * 3  # 分母    
            for i in range(spa_row_before, spa_row_after):
                for j in range(spa_col_before, spa_col_after):
                    if LC_Value[i][j] == lc_type and spa_cu_dataset[i][j] <= 70:
                        euclideanDis = math.sqrt(math.pow((pos[0] - i), 2) + math.pow((pos[1] - j), 2))
                        if euclideanDis != 0 : euclideanDis = math.pow(euclideanDis, -euc_pow)
                        # print(euclideanDis)
              # 在欧氏距离的基础上再按照MQC比重分配
                        numerator[0] += (euclideanDis * spa_cu_before_dataset[i][j] * MQC_Score_before[i][j] * 0.1)
                        numerator[1] += (euclideanDis * spa_cu_dataset[i][j] * MQC_Score[i][j] * 0.1)
                        numerator[2] += (euclideanDis * spa_cu_after_dataset[i][j] * MQC_Score_after[i][j] * 0.1)
                        denominator[0] += euclideanDis * MQC_Score_before[i][j] * 0.1
                        denominator[1] += euclideanDis * MQC_Score[i][j] * 0.1
                        denominator[2] += euclideanDis * MQC_Score_after[i][j] * 0.1
                        
          # 当n*n范围内无相同lc时，使用原始值填充
            if denominator[0] > 0:
                spa_result_inter = round(numerator[1]/denominator[1])
                before_weight = abs((numerator[0]/denominator[0]) - or_before_value)
                after_weight = abs((numerator[2]/denominator[2]) - or_after_value)
                spa_weight = round((before_weight + after_weight) / 2, 2)
                oneSizeWeight_arr.append(spa_weight)
            else : 
                oneSizeWeight_arr.append(0)
                spa_result_inter = or_value
                # print('eq zero', winsize, pos)

            one_O_value.append(or_value)
            one_F_value.append(spa_result_inter)

        O_value.append(one_O_value)
        F_value.append(one_F_value)
        w_array.append(oneSizeWeight_arr) 

    # print(O_value) 
    # print(F_value)

    result = calculatedif(O_value, F_value, fill_pos_length)
    
    return result

    #绘制权重变化折线图
    # line_arr = []
    # for index in range(0, 100):
    #     elem_arr = []
    #     for ele in w_array:
    #         elem_arr.append(ele[index])
    #     line_arr.append(elem_arr)
  
    # print(line_arr)

    # pic_ind = 1
    # for ele in line_arr:
    #     # if np.mean(ele) > 10 :
    #         aa = np.arange(2, 11, 1)
    #         plt.figure(figsize=(10, 5)) #宽，高
    #         plt.plot(aa, ele, label='count', color='#fd7400',  marker='o', markersize=5)
    #         plt.xlabel('Size', fontsize=15, family='Times New Roman') 
    #         plt.ylabel('Weight', fontsize=15, family='Times New Roman')
    #         plt.savefig("./Spatial_W2_Result/MQC_lt_30/random1/1116_%d" % pic_ind, dpi=300)
    #         pic_ind +=1
    #         plt.show()
        # else: pic_ind +=1

#calculate MRE and RMSE
def calculatedif (O_value, F_value, fill_pos_length):
    MRE = []
    RMSE = []
    for i in range(0, 20):
        numera_mre = 0 
        denomin = 0
        numera_rmse = 0
        for j in range(0, fill_pos_length):
            v_mre = abs(O_value[i][j] - F_value[i][j])
            v_rmse = math.pow((O_value[i][j] - F_value[i][j]), 2)
            numera_mre += v_mre
            denomin += O_value[i][j]
            numera_rmse += v_rmse
        MRE.append(round(numera_mre / denomin, 3))
        RMSE.append(round(math.sqrt(numera_rmse / fill_pos_length), 3))
    
    # print(MRE)
    # print(RMSE)
    return {'MRE': MRE, 'RMSE': RMSE}
    # aa = np.arange(1, 21, 1)
    # plt.figure(figsize=(10, 5)) #宽，高
    # plt.plot(aa, MRE, label='count', color='#fd7400',  marker='o', markersize=5)
    # plt.title('MRE_Power_%d' % euc_pow, fontsize=18, family='Times New Roman')
    # plt.xlabel('Size', fontsize=15, family='Times New Roman') 
    # plt.ylabel('MRE', fontsize=15, family='Times New Roman')
    # plt.savefig("./Spatial_W2_Result/MRE/power_%d" % euc_pow, dpi=300)
    # plt.show()

    # aa = np.arange(1, 21, 1)
    # plt.figure(figsize=(10, 5)) #宽，高
    # plt.plot(aa, RMSE, label='count', color='#fd7400',  marker='o', markersize=5)
    # plt.title('RMSE_Power_%d' % euc_pow, fontsize=18, family='Times New Roman')
    # plt.xlabel('Size', fontsize=15, family='Times New Roman') 
    # plt.ylabel('RMSE', fontsize=15, family='Times New Roman')
    # plt.savefig("./Spatial_W2_Result/RMSE/power_%d" % euc_pow, dpi=300)
    # plt.show()

def render_MQC (MQC_Score, title='MQC'):
    plt.imshow(MQC_Score, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    plt.show()

def render_Img (data):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title('Image', family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
    plt.show()

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
    # plt.savefig(url, dpi=300)
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

fileLists = ReadDirFiles.readDir('../HDF/h11v04')
# print('lists', len(fileLists))

fileDatas = []
count = 20
for file in fileLists:
  fileDatas.append(ReadFile(file))


LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h11v04.006.2019199203448.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集

LC_info = gdal.Open(LC_subdatasets[0][0]).ReadAsArray()


# read MQC file
MQC_All = np.load('./MQC_NP/h11v04_2018.npy')

fileIndex = 8

# print(ds)
# render_MQC(MQC_All[fileIndex - 1])
# render_Img(fileDatas[fileIndex])
# print(MQC_Score)
# MQC_part = []
# for i in range(0, 500):
#     row = []
#     for j in range(700, 1200):
#         row.append(MQC_Score[i][j])
#     MQC_part.append(row)
# print(MQC_part)
# print(np.mean(array(MQC_part,'f').flatten()))
# render_MQC(MQC_part, 'MQC_Part')


fill_pos_length = 500


all_great_pos = get_GreatPixel (MQC_All[fileIndex - 1], fileDatas[fileIndex])
# print(len(all_great_pos))

rand_pos = int_random(500, 300000, fill_pos_length)
fill_pos = []
for ele in rand_pos:
    fill_pos.append(all_great_pos[ele])

line_array = []
for euc_pow in range(1, 6):
    result = Spatial_W2(fileDatas, fileIndex, fill_pos, LC_info, MQC_All, euc_pow, fill_pos_length)
    line_array.append(result)


draw_multiLine (line_array, 'MRE', './Spatial_W2_Result/MRE/Spatial_line_1209')
draw_multiLine (line_array, 'RMSE', './Spatial_W2_Result/RMSE/Spatial_line_1209')