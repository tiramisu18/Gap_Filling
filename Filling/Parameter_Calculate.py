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
import copy
import ReadDirFiles
import math
import h5py
import time
import random
import Filling_Pixel
import Draw_PoltLine

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
