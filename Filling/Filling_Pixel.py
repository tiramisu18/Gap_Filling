import os
from typing import MappingView
import numpy as np
from numpy.random.mtrand import sample
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import copy
import math
import time

def Temporal_Cal (fileDatas, index, Filling_Pos, LC_info, MQC_File, temporalLength, tem_winSize_unilateral, SES_pow):
    # print('begin_tem', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  # interpolation
    cu_dataset = fileDatas[index]
    # temporalLength =  6  
    tem_filling_value = 0
    tem_weight = 0
    tem_back_index = index + temporalLength
    tem_forward_index = index - temporalLength
    if index + temporalLength > len(fileDatas) - 1 : tem_back_index = len(fileDatas) - 1
    if index - temporalLength < 0 : tem_forward_index = -1
    # tem_winSize_unilateral = 2  # n*2 + 1
     
    pos = Filling_Pos

    lc_type = LC_info[pos[0]][pos[1]] 
    or_value = cu_dataset[pos[0]][pos[1]]    
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
            if LC_info[i][j] == lc_type:
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
                    if(i == pos[0] and j == pos[1]): tem_filling_value = round(inter)
                    else :
                        dif_value = abs(inter - cu_dataset[i][j])
                        tem_wei_count += dif_value
                else: 
                    tem_filling_value = or_value
                    # print('eq 0 ', i, valid_lc, tem_filling_value)
                tem_index += 1
    if valid_lc == 0 :
        tem_weight = 0 
    else :
        tem_weight = (round(tem_wei_count/valid_lc, 2)) 
    # print('end_tem', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
   
    return {'weight': tem_weight, 'filling': tem_filling_value, 'or_value': or_value}

def Spatial_Cal (fileDatas, index, Filling_Pos, LC_info, MQC_File, EUC_pow, spa_winSize_unilateral):
    # print('begin_spa', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    spa_filling_value = 0
    spa_weight = 0
    spa_cu_dataset = fileDatas[index]
    spa_cu_before_dataset = fileDatas[index - 1]
    spa_cu_after_dataset = fileDatas[index + 1]
    # spa_winSize_unilateral = 10 # n*2 + 1

    MQC_Score = MQC_File[index - 1]
    MQC_Score_before = MQC_File[index - 2]
    MQC_Score_after = MQC_File[index]
    pos = Filling_Pos  

    lc_type = LC_info[pos[0]][pos[1]]    
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
            if LC_info[i][j] == lc_type and spa_cu_dataset[i][j] <= 70:
                euclideanDis = math.sqrt(math.pow((pos[0] - i), 2) + math.pow((pos[1] - j), 2))
                if euclideanDis != 0 : euclideanDis = math.pow(euclideanDis, -EUC_pow)
                # 在欧氏距离的基础上再按照MQC比重分配
                numerator[0] += (euclideanDis * spa_cu_before_dataset[i][j] * MQC_Score_before[i][j] * 0.1)
                numerator[1] += (euclideanDis * spa_cu_dataset[i][j] * MQC_Score[i][j] * 0.1)
                numerator[2] += (euclideanDis * spa_cu_after_dataset[i][j] * MQC_Score_after[i][j] * 0.1)
                denominator[0] += euclideanDis * MQC_Score_before[i][j] * 0.1
                denominator[1] += euclideanDis * MQC_Score[i][j] * 0.1
                denominator[2] += euclideanDis * MQC_Score_after[i][j] * 0.1
                        
    # 当n*n范围内无相同lc时，使用原始值填充
    if denominator[0] > 0:
        spa_filling_value = round(numerator[1]/denominator[1])
        before_weight = abs((numerator[0]/denominator[0]) - or_before_value)
        after_weight = abs((numerator[2]/denominator[2]) - or_after_value)
        spa_weight = round((before_weight + after_weight) / 2, 2)
    else : 
        spa_filling_value = or_value
        # print('eq zero', winsize, pos)


    # print('end_spa', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
   
    return {'weight': spa_weight, 'filling': spa_filling_value, 'or_value': or_value}

def Fill_Pixel (fileDatas, index, Filling_Pos, LC_info, MQC_File, temporalLength, tem_winSize_unilateral, SES_pow, EUC_pow, spa_winSize_unilateral):
 
    # LAI_Result = copy.deepcopy(fileDatas[index])
    # interpolation

    Fil_tem = []
    Fil_spa = []
    Fil_value = []
    Tem_W = []
    Spa_W = []

    for pos in Filling_Pos:    
        tem_ob = Temporal_Cal (fileDatas, index, pos, LC_info, MQC_File, temporalLength, tem_winSize_unilateral, SES_pow)
        spa_ob = Spatial_Cal (fileDatas, index, pos, LC_info, MQC_File, EUC_pow, spa_winSize_unilateral) 
        MQC_value = MQC_File[index - 1][pos[0]][pos[1]]  
        or_value = fileDatas[index][pos[0]][pos[1]] 
        final = or_value
        spa_filling_value = spa_ob['filling']
        spa_weight = spa_ob['weight']
        tem_filling_value = tem_ob['filling']
        tem_weight = tem_ob['weight']

        Fil_tem.append(tem_filling_value)
        Fil_spa.append(spa_filling_value)
        Tem_W.append(tem_weight)
        Spa_W.append(spa_weight)
        # total Calculation
        final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight)) 
        # if (MQC_value >= 30):        
        #     final = round((or_value * MQC_value * 0.1 + spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (MQC_value * 0.1 + spa_weight + tem_weight))  
        # else :
        #     if spa_weight != 0 and tem_weight != 0 : 
        #         final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight))  
        
        Fil_value.append(final)

    # print({'tem': Fil_tem, 'spa': Fil_spa, 'Fil': Fil_value})      
    # LAI_Result[pos[0]][pos[1]] = final
    return {'Tem': Fil_tem, 'Spa': Fil_spa, 'Fil': Fil_value, 'T_W': Tem_W, 'S_W': Spa_W}

def Fill_Pixel_MQCPart (fileDatas, index, Filling_Pos, LC_info, MQC_File, temporalLength, tem_winSize_unilateral, SES_pow, EUC_pow, spa_winSize_unilateral, method):
 
    # LAI_Result = copy.deepcopy(fileDatas[index])
    # interpolation

    Fil_tem = []
    Fil_spa = []
    Fil_value = []
    Tem_W = []
    Spa_W = []
    Mqc_W = []

    for pos in Filling_Pos:    
        tem_ob = Temporal_Cal (fileDatas, index, pos, LC_info, MQC_File, temporalLength, tem_winSize_unilateral, SES_pow)
        spa_ob = Spatial_Cal (fileDatas, index, pos, LC_info, MQC_File, EUC_pow, spa_winSize_unilateral) 
        MQC_value = MQC_File[index - 1][pos[0]][pos[1]]  
        or_value = fileDatas[index][pos[0]][pos[1]] 
        final = or_value
        spa_filling_value = spa_ob['filling']
        spa_weight = spa_ob['weight']
        tem_filling_value = tem_ob['filling']
        tem_weight = tem_ob['weight']

        Fil_tem.append(tem_filling_value)
        Fil_spa.append(spa_filling_value)
        Tem_W.append(tem_weight)
        Spa_W.append(spa_weight)
        Mqc_W.append(MQC_value * 0.1)
        # total Calculation
        # final = round((or_value * MQC_value * 0.1 + spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (MQC_value * 0.1 + spa_weight + tem_weight))  
        if method == 1 :
            final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight))  
        else :
            if (MQC_value >= 10):        
                final = round((or_value * MQC_value * 0.1 + spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (MQC_value * 0.1 + spa_weight + tem_weight))  
            else :
                if spa_weight != 0 and tem_weight != 0 : 
                    final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight))  
        
        Fil_value.append(final)

    # print({'tem': Fil_tem, 'spa': Fil_spa, 'Fil': Fil_value})      
    # LAI_Result[pos[0]][pos[1]] = final
    return {'Tem': Fil_tem, 'Spa': Fil_spa, 'Fil': Fil_value, 'T_W': Tem_W, 'S_W': Spa_W, 'M_W': Mqc_W}
