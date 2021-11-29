#  探究时间相关性中的参数最佳值
import os
from typing import MappingView
import numpy as np
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

def Temporal_w3 (fileDatas, index, position_nan, LC_Value, MQC_File, SES_pow, temporalLength):
    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    # Filling_Pos = []
    # Filling_Pos = position_nan[1000600: 1000700]
    Filling_Pos = position_nan
  # interpolation
    spa_cu_dataset = fileDatas[index]
    # temporalLength =  6  #len(fileDatas)
    tem_result_inter = 0
    tem_back_index = index + temporalLength
    tem_forward_index = index - temporalLength
    if index + temporalLength > len(fileDatas) : tem_back_index = len(fileDatas)
    if index - temporalLength < 0 : tem_forward_index = -1
    # tem_winSize_unilateral = 2  # n*2 + 1

    w_array = []
    F_value = []
    O_value = []
    for size in range(3, 23):
        tem_winSize_unilateral = size
        oneSize_arr = []
        one_F_value = []
        one_O_value = []    
        for pos in Filling_Pos:    
            lc_type = LC_Value[pos[0]][pos[1]] 
            or_value = spa_cu_dataset[pos[0]][pos[1]]    
            numerator = [] # 分子
            denominator = []  # 分母
            tem_index = 0
            tem_wei_count = 0
            tem_row_before = 0 
            tem_row_after = 2400
            tem_col_before = 0 
            tem_col_after = 2400
            valid_lc = 0

            if pos[0]- tem_winSize_unilateral > 0 : tem_row_before = pos[0]- tem_winSize_unilateral
            if pos[0] + tem_winSize_unilateral < 2400 : tem_row_after = pos[0] + tem_winSize_unilateral
            if pos[1]- tem_winSize_unilateral > 0 : tem_col_before = pos[1]- tem_winSize_unilateral
            if pos[1] + tem_winSize_unilateral < 2400 : tem_col_after = pos[1] + tem_winSize_unilateral
    
            for i in range(tem_row_before, tem_row_after):
                for j in range(tem_col_before, tem_col_after):
                    if LC_Value[i][j] == lc_type:
                        forward_index = index - 1
                        backward_index = index + 1
                        forward_i = 1
                        backward_i = 1
                        numerator.append(0)
                        denominator.append(0)
                        while (forward_index > tem_forward_index):
                            value = fileDatas[forward_index][i][j]
                            tem_SES = SES_pow * math.pow((1 - SES_pow), forward_i - 1)
                            if(value <= 70):
                                MQC_Score = MQC_File[forward_index - 1] 
                                numerator[tem_index] += value * tem_SES * MQC_Score[i][j] * 0.1
                                denominator[tem_index] += tem_SES * MQC_Score[i][j] * 0.1                                
                            forward_index -= 1
                            forward_i += 1
                        while (backward_index < tem_back_index):
                            value = fileDatas[backward_index][i][j]
                            tem_SES = SES_pow * math.pow((1 - SES_pow), backward_i - 1)
                            if(value <= 70):
                                MQC_Score = MQC_File[backward_index - 1]
                                numerator[tem_index] += value * tem_SES * MQC_Score[i][j] * 0.1
                                denominator[tem_index] += tem_SES * MQC_Score[i][j] * 0.1 
                            backward_index += 1
                            backward_i += 1
                        if denominator[tem_index] != 0 : 
                            valid_lc +=1       
                            inter = numerator[tem_index] / denominator[tem_index]
                            if(i == pos[0] and j == pos[1]): tem_result_inter = round(inter)
                            else :
                                dif_value = abs(inter - spa_cu_dataset[i][j])
                                tem_wei_count += dif_value
                        else: 
                            tem_result_inter = or_value
                            # print('eq 0 ', i, valid_lc, tem_result_inter)
                        tem_index += 1
            # print(spa_weight, tem_weight)
            oneSize_arr.append(round(tem_wei_count/valid_lc, 2)) 
            one_O_value.append(or_value)
            one_F_value.append(tem_result_inter)

        O_value.append(one_O_value)
        F_value.append(one_F_value)
        w_array.append(oneSize_arr) 
    # print(O_value, len(O_value[0])) 
    # print(F_value, len(F_value[0]))

    result = calculatedif(O_value, F_value)   
    return result

#calculate MRE and RMSE
def calculatedif (O_value, F_value):
    MRE = []
    RMSE = []
    for i in range(0, 20):
        numera_mre = 0 
        denomin = 0
        numera_rmse = 0
        for j in range(0, 100):
            v_mre = abs(O_value[i][j] - F_value[i][j])
            v_rmse = math.pow((O_value[i][j] - F_value[i][j]), 2)
            numera_mre += v_mre
            denomin += O_value[i][j]
            numera_rmse += v_rmse
        MRE.append(round(numera_mre / denomin, 3))
        RMSE.append(round(math.sqrt(numera_rmse / len(O_value)), 3))
    
    # print(MRE)
    # print(RMSE)
    return {'MRE': MRE, 'RMSE': RMSE}

def draw_multiLine (data, key, url, state):
    aa = np.arange(3, 23, 1)
    # plt.xticks(aa)
    plt.figure(figsize=(15, 6)) #宽，高
    # plt.title('MRE', family='Times New Roman', fontsize=18)   
    plt.xlabel('WinSize', fontsize=15, family='Times New Roman') 
    plt.ylabel(key, fontsize=15, family='Times New Roman')
    line1=plt.plot(aa,data[0][key], label='count', color='#1f8a6f',  marker='o', markersize=5)
    line2=plt.plot(aa,data[1][key], label='count', color='#bfdb39',  marker='o', markersize=5)
    line3=plt.plot(aa,data[2][key], label='count', color='#016382',  marker='o', markersize=5)
    line4=plt.plot(aa,data[3][key], label='count', color='#fd7400',  marker='o', markersize=5)
    line5=plt.plot(aa,data[4][key], label='count',color='#ffe117',  marker='o', markersize=5)
    if state == 1 :
        plt.legend(
        (line1[0],  line2[0],  line3[0],  line4[0],  line5[0]),     
        ('Pow_0.3', 'Pow_0.35','Pow_0.4','Pow_0.45','Pow_0.5'),
        loc = 1, prop={'size':15, 'family':'Times New Roman'},
        )
    else :
        plt.legend(
        (line1[0],  line2[0],  line3[0],  line4[0],  line5[0]),     
        ('Length_4','Length_5','Length_6','Length_7','Length_8'),
        loc = 1, prop={'size':15, 'family':'Times New Roman'},
        )
    # plt.savefig(url, dpi=300)
    plt.show()


fileLists = ReadDirFiles.readDir(
  'C:\JR_Program\Filling_Missing_Values\h27v06')
# print('lists', len(fileLists))

fileDatas = []
count = 20
for file in fileLists:
  fileDatas.append(ReadFile(file))


LC_file = gdal.Open('C:\JR_Program\Filling_Missing_Values\LC\MCD12Q1.A2018001.h27v06.006.2019200015326.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集

LC_info = gdal.Open(LC_subdatasets[0][0]).ReadAsArray()


# read MQC file
path='./MQC_files/h27v06_2018_MQC_Score.mat'             
MQC_File=h5py.File(path) 
# print(MQC_File.keys())
file_Content = MQC_File["MQC_Score"]


print('1', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
MQC_All = []
for idx in range(0, 44):
    MQC_data = MQC_File[file_Content[0,idx]]  # [column, row]
    # MQC_Score = np.transpose(MQC_data[:])
    MQC_Score = MQC_data[:]
    MQC_All.append(MQC_Score)
print(len(MQC_All))
print('2', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

fileIndex = 12

# rand_i = [313, 307, 204, 370, 372, 210, 361, 433, 331, 331, 351, 403, 340, 197, 175, 377, 372, 164, 23, 337, 213, 36, 350, 50, 396, 147, 22, 336, 103, 236, 60, 263, 330, 156, 388, 412, 217, 90, 444, 246, 16, 261, 212, 264, 181, 376, 342, 27, 478, 67, 208, 424, 457, 168, 177, 469, 117, 83, 21, 209, 399, 497, 353, 29, 402, 373, 488, 272, 222, 96, 137, 231, 84, 1, 354, 286, 81, 15, 318, 499, 491, 184, 72, 474, 454, 466, 482, 239, 78, 284, 260, 438, 195, 455, 390, 148, 347, 448, 248, 429]
# rand_j = [939, 940, 717, 1050, 839, 818, 1080, 956, 935, 702, 705, 1117, 936, 830, 1140, 942, 1160, 1057, 752, 898, 1027, 849, 736, 719, 778, 846, 1116, 881, 1043, 759, 800, 1131, 1191, 1001, 1194, 1192, 1150, 1186, 810, 746, 715, 997, 744, 1042, 1005, 804, 1025, 1073, 730, 966, 1147, 845, 1023, 833, 911, 718, 841, 878, 1033, 1035, 775, 991, 1020, 733, 1067, 1144, 975, 938, 728, 1009, 843, 869, 1056, 1039, 1159, 861, 1155, 1108, 969, 1058, 1196, 956, 769, 1128, 959, 703, 883, 921, 978, 704, 1100, 847, 1159, 984, 1012, 823, 1156, 1142, 1115, 835]
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
fill_pos_mqc = []
fill_pos_val = []
fill_pos = []
for i in range(0, 100):
    try :
        fill_pos.append([rand_i[i], rand_j[i]])
        fill_pos_mqc.append(MQC_Score[rand_i[i]][rand_j[i]])
        fill_pos_val.append(fileDatas[fileIndex][rand_i[i]][rand_j[i]])
    except:
        print(i)

# print(fill_pos_mqc)
# print(fill_pos_val)

# SES_pow_array = [0.3, 0.35, 0.4, 0.45, 0.5]
# line_array = []
# for SES_pow in SES_pow_array:
#     result = Temporal_w3(fileDatas, fileIndex, fill_pos, LC_info, MQC_All, SES_pow, 6)
#     line_array.append(result)

# draw_multiLine (line_array, 'MRE', './Spatial_W2_Result/MRE/Temporal_line_1123_length_4', 1) 
# draw_multiLine (line_array, 'RMSE', './Spatial_W2_Result/RMSE/Temporal_line_1123_length_4', 1)

line_array = []
for length in range(4, 9):
    result = Temporal_w3(fileDatas, fileIndex, fill_pos, LC_info, MQC_All, 0.35, length)
    line_array.append(result)

draw_multiLine (line_array, 'MRE', './Spatial_W2_Result/MRE/Temporal_line_1123_length_4', 2)
draw_multiLine (line_array, 'RMSE', './Spatial_W2_Result/RMSE/Temporal_line_1123_length_4', 2)


# for file_index in range(2, 3):
#     print(file_index)
#     MQC_Score = MQC_All[file_index - 1]
#     oneSample = fileDatas[file_index]
#     Filling_Pos = []
#   # data_one = copy.deepcopy(MQC_Score)
#     for i in range(2400):
#         for j in range(2400):
#             if MQC_Score[i][j] < 35 and oneSample[i][j] <= 70 : 
#                 Filling_Pos.append([i, j])
#                 # data_one[i][j] = 'nan'

#     print(len(Filling_Pos))
#     Temporal_w3(fileDatas, file_index, Filling_Pos, LC_info, MQC_All, 0.5)




