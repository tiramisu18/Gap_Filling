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
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC}

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


def render_QC (QC_data, title='Algo Path', issave=False, savepath=''):
    plt.imshow(QC_data, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()
    # colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    # bounds = [-1,0,5,10]
    # cmap = pltcolor.ListedColormap(colors)
    # norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    # plt.title(title, family='Times New Roman', fontsize=18)
    # plt.imshow(QC_data, cmap=cmap, norm=norm)
    # cbar = plt.colorbar()
    # cbar.set_ticklabels(['-1','0','5','10'])
    # plt.show()

def render_Img (data, title='Image', issave=False, savepath=''):
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

# 读取二进制QC的算法路径，转为相应的权重
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

# 读取二进制QC的云信息，转为相应的权重
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

#calculate MRE and RMSE
def calculatedif (O_value, F_value, first_length, fill_pos_length):
    MRE = []
    RMSE = []
    for i in range(0, first_length):
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

# 求权重的最佳值 
def get_wight_better_para(QC_All, fileIndex, fileDatas, LC_info, type):
    # Spatial
    if type == 1:
        pos_count = 500
        Filling_Pos = random_pos(QC_All[fileIndex], 2000, pos_count)
        print(len(Filling_Pos))
        winsi_len = 11
        line_array = []
        for euc_pow in range(1, 6):
            print(euc_pow)
            pow_one_or = []
            pow_one_fil = []
            pow_one_we = []
            for win_size in range(3, winsi_len):
                # re = Filling_Pixel.Fill_Pixel(fileDatas, fileIndex, Filling_Pos, LC_info, QC_All, 6, 12, 0.35, euc_pow, win_size)
                re = Filling_Pixel.Fill_Pixel_One(fileDatas, fileIndex, Filling_Pos, LC_info, QC_All, 6, 12, 0.35, euc_pow, win_size, 2)
                pow_one_or.append(re['Or'])
                pow_one_fil.append(re['Fil'])
                pow_one_we.append(round(np.mean(re['Weight']), 3))
            line_array.append(pow_one_we)
            # result = calculatedif(pow_one_or, pow_one_fil, winsi_len-1, len(Filling_Pos))
            # line_array.append(result['RMSE'])
        Draw_PoltLine.draw_polt_Line(np.arange(3, winsi_len, 1),{
            'title': 'Count_%d' % pos_count,
            'xlable': 'Half Width',
            'ylable': 'Weight',
            'line': line_array,
            'le_name': ['Pow=1', 'Pow=2', 'Pow=3', 'Pow=4', 'Pow=5'],
            'color': False,
            'marker': False,
            'lineStyle': []
            },'./Daily_cache/0126/0126_Spa_%s_Count_%d'% (fileIndex, pos_count), True, 1)
    # Temporal
    else:
        pos_count = 50
        Filling_Pos = random_pos(QC_All[fileIndex], 2000, pos_count)
        # print(len(Filling_Pos))
        winsi_len = 10
        line_array = []
        SES_pow_array = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for ses_pow in SES_pow_array:
            print(ses_pow)
            pow_one_or = []
            pow_one_fil = []
            pow_one_we = []
            for win_size in range(2, winsi_len):
                re = Filling_Pixel.Fill_Pixel_One(fileDatas, fileIndex, Filling_Pos, LC_info, QC_All, 6, win_size, ses_pow, 2, 5, 1)
                pow_one_or.append(re['Or'])
                pow_one_fil.append(re['Fil'])
                pow_one_we.append(round(np.mean(re['Weight']), 3))
            line_array.append(pow_one_we)
        # print(line_array)
            # result = calculatedif(pow_one_or, pow_one_fil, winsi_len-5, len(Filling_Pos))
            # line_array.append(result['RMSE'])
        Draw_PoltLine.draw_polt_Line(np.arange(2, winsi_len, 1),{
            'title': 'Count_%d' % pos_count,
            'xlable': 'Half Width',
            'ylable': 'Weight',
            'line': line_array,
            'le_name': ['Pow=0.2', 'Pow=0.3','Pow=0.4', 'Pow=0.5', 'Pow=0.6', 'Pow=0.7'],
            'color': False,
            'marker': False,
            'lineStyle': []
            },'./Daily_cache/0126/0126_Tem_%s_Count_%d'% (fileIndex, pos_count), False, 1)

# 模拟数据的时空填补
def Simu_filling(x_v, y_v):
    LAI_Simu_noErr = np.load('../Validation/Simulation_Dataset/LAI_Simu_noErr.npy')
    # LAI_Simu_addErr = np.load('../Validation/Simulation_Dataset/LAI_Ori.npy')
    LAI_Simu_addErr = np.load('../Validation/Simulation_Dataset/LAI_Simu_addErr.npy')
    LandCover = np.load('../Validation/Simulation_Dataset/LandCover.npy')
    Err_weight= np.load('../Validation/Simulation_Dataset/Err_weight.npy')
    
    Filling_Pos = [[x_v, y_v]]
    Fil_val_1 = []
    Fil_val_2 = []
    Fil_val = []
    ori_val = []
    simu_val = []

    ses_pow = 0.8
    for index in range(1, 45):
        # if index < 10: ses_pow = 0.3
        # if 10 < index < 14 or 36 < index < 40: ses_pow = 0.5
        # elif 14 < index < 20 or 30 < index < 36: ses_pow = 0.8
        # else: ses_pow = 0.3
        # elif 20 < index < 30: ses_pow = 0.3
        re1 = Filling_Pixel.Fill_Pixel(LAI_Simu_addErr, index, Filling_Pos, LandCover, Err_weight, 6, 12, ses_pow, 2, 5)
        # re1 = Filling_Pixel.Fill_Pixel_noQC(LAI_Simu_addErr, index, Filling_Pos, LandCover, 6, 12, ses_pow, 2, 5)
        Fil_val_1.append(re1['Tem'][0]/10)
        Fil_val_2.append(re1['Spa'][0] /10)
        Fil_val.append(re1['Fil'][0] /10)
        ori_val.append(re1['Or'][0] /10)
        simu_val.append(LAI_Simu_noErr[index][x_v][y_v])
   
    Draw_PoltLine.draw_polt_Line(np.arange(1, 45, 1),{
        'title': 'pos_%s_%s' % (x_v, y_v),
        'xlable': 'Day',
        'ylable': 'LAI',
        # 'line': [ori_val, Fil_val, Fil_val_1, Fil_val_2],
        'line': [simu_val, ori_val, Fil_val],
        'le_name': ['Simu','Original', 'Filling','Temporal', 'Spatial'],
        'color': ['gray', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
        'marker': False,
        'lineStyle': ['dashed'],
        },'./Daily_cache/0315/Filling', True, 2)

Simu_filling(55, 55)

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

# print(QCDatas[fileIndex])
# for i in range(0, 46):
#     render_QC(QC_All[i], '2018_%02d' %(i * 8 + 1), True, '../QC/Img/h11v04_2018/h11v04_2018_%d' %(i+1))
#     render_Img(fileDatas[i], '2018_%02d' %(i * 8 + 1), True, '../Original_Data/PNG/h11v04_2018/h11v04_2018_%d' %(i+1))
#     render_QC(QC_Cloud[i], '2018_%02d' %(i * 8 + 1), True, '../QC/Img/h11v04_2018_CloudState/h11v04_2018_%d' %(i+1))

# render_Img(fileDatas[fileIndex])
# print(LC_info[800][800], LC_info[1200][1200])

x_v = 1200
y_v = 1200
pixel_data = []
pixel_pos = {'x': x_v, 'y': y_v}
for i in range(1, 45):
    pixel_val = fileDatas[i][pixel_pos['x']][pixel_pos['y']] / 10
    pixel_data.append(pixel_val)

Filling_Pos = [[x_v, y_v]]
Fil_val_1 = []
Fil_val_2 = []
Fil_val_3 = []
Tem_W = []
Spa_W = []
Qc_W = []

ses_pow = 0.8
for index in range(1, 45):
    if index < 10: ses_pow = 0.3
    if 10 < index < 14 or 36 < index < 40: ses_pow = 0.5
    elif 14 < index < 20 or 30 < index < 36: ses_pow = 0.8
    else: ses_pow = 0.3
    # elif 20 < index < 30: ses_pow = 0.3

    re1 = Filling_Pixel.Fill_Pixel(fileDatas, index, Filling_Pos, LC_info, QC_All, 6, 12, ses_pow, 2, 5)
    # re1 = Filling_Pixel.Fill_Pixel_MQCPart(fileDatas, index, Filling_Pos, LC_info, MQC_All, 6, 12, 0.35, 2, 6, 2) # no MQC
    Fil_val_1.append(re1['Tem'][0] / 10)
    Fil_val_2.append(re1['Spa'][0] / 10)
    Fil_val_3.append(re1['Fil'][0] / 10)
    Tem_W.append(re1['T_W'][0])
    Spa_W.append(re1['S_W'][0])
    Qc_W.append(re1['Qc_W'][0])
# print(Fil_val)
# Draw_PoltLine.draw_Line(np.arange(2, 43, 1),pixel_data, Fil_val_1, Fil_val_2, Fil_val_3, './Daily_cache/pos_%s_%s_indep_part' % (x_v, y_v), False, 'pos_%s_%s_indep_part' % (x_v, y_v))

Draw_PoltLine.draw_polt_Line(np.arange(1, 45, 1),{
    'title': 'pos_%s_%s' % (x_v, y_v),
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': [pixel_data, Fil_val_1],
    'le_name': ['Original', 'Temporal'],
    'color': ['gray', '#bfdb39'],
    'marker': False,
    'lineStyle': ['dashed'],
    },'./Daily_cache/1215_4', False, 2)


# Draw_PoltLine.draw_polt_Line(np.arange(1, 45, 1),{
#     'title': 'pos_%s_%s' % (x_v, y_v),
#     'xlable': 'Day',
#     'ylable': 'LAI',
#     'line': [pixel_data, Fil_val_1, Fil_val_2, Fil_val_3],
#     'le_name': ['Original', 'Tem', 'Spa', 'Fil'],
#     'color': ['gray', '#bfdb39', '#ffe117', '#fd7400', '#1f8a6f', '#548bb7'],
#     'marker': False,
#     'lineStyle': ['dashed']
#     },'./Daily_cache/pos_%s_%s_0125' % (x_v, y_v), True, 2)

# Draw_PoltLine.polt_Line_twoScale(np.arange(1, 45, 1),{
#     'title': 'pos_%s_%s_w' % (x_v, y_v),
#     'xlable': 'Day',
#     'ylable': 'LAI',
#     'line': [[pixel_data], [Tem_W, Spa_W, Qc_W]],
#     'le_name': ['Original', 'Tem_W', 'Spa_W', 'Qc_W'],
#     'color': ['gray', '#bfdb39', '#ffe117', '#fd7400', ],
#     'marker': False,
#     'lineStyle': ['dashed']
#     },'./Daily_cache/0125/pos_%s_%s_0125_w' % (x_v, y_v), True, 2)


