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
    
    # random_pos = [1357014, 64580, 1614800, 1475407, 1437849, 1268490, 1906901, 1779502, 1269123, 1218517, 438127, 1983877, 666523, 382462, 953449, 1664506, 26541, 164652, 340312, 942213, 1120944, 1917218, 595743, 1428464, 1146849, 674744, 653850, 1732185, 649105, 1857483, 1651886, 921660, 1355108, 1672042, 1758674, 1528441, 770445, 264914, 169181, 591573, 1342472, 950553, 490852, 991788, 780714, 1191097, 558402, 1623274, 406704, 878968, 55637, 1235597, 440304, 1730134, 13632, 1106432, 951711, 2595, 1098209, 1913674, 1738643, 928997, 836309, 1550047, 1172073, 1097860, 1837628, 1380302, 771129, 1782911, 2054519, 170868, 1551374, 1464954, 1722847, 1752209, 2030930, 428020, 1176203, 1031184, 591186, 3055, 1941129, 1116496, 2025878, 498899, 924412, 512907, 1310214, 894081, 149721, 1959503, 548949, 1013993, 615769, 1556444, 357461, 1836331, 737808, 1532753]
    random_pos_1 = [277571, 1905328, 1636417, 977632, 124750, 1958541, 318085, 284845, 384361, 470948, 179682, 1239124, 1532915, 680389, 325237, 37190, 626692, 423558, 1740277, 1437388, 1307806, 1464623, 1480660, 1010158, 517348, 1978000, 1207013, 98076, 388062, 1475472, 699801, 965017, 863808, 293299, 659766, 1318790, 889336, 1487816, 1233725, 1402682, 1638549, 1437189, 907725, 238432, 1888861, 1009447, 1360904, 385694, 685730, 661973, 61988, 706459, 1093202, 1284857, 1162561, 471960, 1724816, 902455, 1286264, 1903879, 1616841, 600309, 141284, 1333861, 58588, 1504626, 1686220, 394960, 324009, 554291, 929436, 1799999, 191329, 640244, 972325, 1362605, 1164801, 1238637, 1382083, 1149997, 693047, 876139, 1027683, 1507533, 1247670, 123537, 816786, 155581, 1379025, 1578739, 1426787, 540413, 690491, 1339521, 1096674, 1650076, 775817, 1963384, 1634810, 755476]
    random_pos_2 = [117033, 215923, 19545, 178608, 206533, 104448, 493174, 593402, 563877, 264287, 447831, 638355, 421925, 62935, 160771, 369520, 241072, 532718, 507544, 398501, 459725, 149051, 557946, 363302, 398297, 172282, 380515, 137483, 557487, 552165, 497512, 236889, 230470, 142129, 224183, 394895, 55874, 581908, 337548, 183142, 118797, 568549, 280308, 95905, 204570, 126624, 317658, 241895, 316623, 150586, 568195, 307313, 373819, 70780, 424473, 430174, 587716, 71866, 275200, 470197, 364445, 390166, 218550, 130564, 615200, 622796, 168633, 398053, 606943, 533371, 405288, 144614, 567736, 215775, 205195, 551093, 178796, 503244, 406358, 443631, 637063, 488439, 115497, 24476, 371582, 426381, 417515, 545968, 641860, 646749, 74867, 68963, 7700, 347493, 348891, 384069, 417761, 392185, 27832, 172902]
    random_pos_3 = [206394, 307990, 137102, 415495, 500363, 173167, 302351, 76375, 548307, 132180, 45385, 113314, 483739, 253358, 334440, 68939, 511830, 58001, 427444, 73297, 547384, 301866, 533999, 66664, 57869, 92119, 353850, 379532, 551443, 96062, 325448, 463010, 40168, 447657, 462709, 12975, 481284, 68014, 112249, 408549, 488109, 271721, 121429, 217381, 434355, 252382, 413751, 121019, 333641, 239501, 434226, 66017, 75980, 413830, 150403, 38603, 494969, 65852, 34458, 32564, 318178, 446986, 9391, 342480, 428667, 109048, 126448, 498003, 442683, 408301, 98924, 134166, 519576, 124892, 549205, 151564, 17540, 445238, 20380, 92952, 57556, 109349, 84536, 508405, 404719, 481961, 46653, 424207, 498049, 308721, 77585, 426453, 216897, 5288, 327835, 552859, 43429, 256518, 25225, 540808]
    # for ele in random_pos_3:
    #     one_pos = Filling_Pos[ele]
    #     test_posi_nan.append(one_pos)
    # print(len(test_posi_nan))
    spa_result_inter = 0
    spa_weight = 0
    spa_cu_dataset = fileDatas[index]
    spa_cu_before_dataset = fileDatas[index - 1]
    spa_cu_after_dataset = fileDatas[index + 1]

    # spa_winSize_unilateral = 10 # n*2 + 1

    MQC_data = MQC_File[MQC_File["MQC_Score"][0,index - 1]]  # [column, row]
    MQC_Score = np.transpose(MQC_data[:])
    MQC_data_before = MQC_File[MQC_File["MQC_Score"][0,index - 2]]  # [column, row]
    MQC_Score_before = np.transpose(MQC_data_before[:])
    MQC_data_after = MQC_File[MQC_File["MQC_Score"][0,index]]  # [column, row]
    MQC_Score_after = np.transpose(MQC_data_after[:])

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
    plt.savefig(url, dpi=300)
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

fileIndex = 10

ds = MQC_File[file_Content[0, fileIndex - 1]]
MQC_Score = np.transpose(ds[:])
# print(ds)
# render_MQC(MQC_Score)
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
fill_pos_length = 2000
for i in range(0, 10):

    all_great_pos = get_GreatPixel (MQC_Score, fileDatas[fileIndex])
# print(len(all_great_pos))

    rand_pos = int_random(500, 300000, fill_pos_length)
    fill_pos = []
    for ele in rand_pos:
        fill_pos.append(all_great_pos[ele])

# randomPos = random_pos
# rand_i = randomPos['i']
# rand_j = randomPos['j']
# fill_pos_mqc = []
# fill_pos_val = []
# fill_pos = []
# for i in range(0, fill_pos_length):
#     try :
#         value = fileDatas[fileIndex][rand_i[i]][rand_j[i]]
#         # print(value)
#         if (value <= 70):
#             fill_pos.append([rand_i[i], rand_j[i]])
#             fill_pos_mqc.append(MQC_Score[rand_i[i]][rand_j[i]])
#             fill_pos_val.append(fileDatas[fileIndex][rand_i[i]][rand_j[i]])
#     except:
#         print(i)

# # print(fill_pos_mqc, mean(fill_pos_mqc))
# # print(fill_pos_val)
# # print(len(fill_pos_val))

    line_array = []
    for euc_pow in range(1, 6):
        result = Spatial_W2(fileDatas, fileIndex, fill_pos, LC_info, MQC_File, euc_pow, fill_pos_length)
        line_array.append(result)


    draw_multiLine (line_array, 'MRE', './Spatial_W2_Result/MRE/Spatial_line_1125_'+ str(i))
    draw_multiLine (line_array, 'RMSE', './Spatial_W2_Result/RMSE/Spatial_line_1125'+ str(i))

# for file_index in range(2, 3):
#     print(file_index)
#     ds = MQC_File[file_Content[0,file_index - 1]]
#     MQC_Score = np.transpose(ds[:])
#     oneSample = fileDatas[file_index]
#     Filling_Pos = []
#   # data_one = copy.deepcopy(MQC_Score)
#     for i in range(2400):
#         for j in range(2400):
#             if MQC_Score[i][j] < 45 and oneSample[i][j] <= 70 : 
#                 Filling_Pos.append([i, j])
#                 # data_one[i][j] = 'nan'

#     print(len(Filling_Pos))
    # Spatial_W2(fileDatas, file_index, Filling_Pos, LC_info, MQC_File)




