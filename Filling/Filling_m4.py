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

# 整个tile填补+时空反距离加权时考虑MQC比重
def Fill_m4 (fileDatas, index, position_nan, LC_Value, MQC_File):
  print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  # colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7', '#d7efb3', '#a57d78', '#8e8681']
  # bounds = [0,10,20,30,40,50,60,70,250]
  # cmap = pltcolor.ListedColormap(colors)
  # norm = pltcolor.BoundaryNorm(bounds, cmap.N)
  # plt.title('Original Image', family='Times New Roman', fontsize=18)
  # plt.imshow(fileDatas[index], cmap=cmap, norm=norm)
  # cbar = plt.colorbar()
  # cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
  # plt.show()

  LAI_Result = copy.deepcopy(fileDatas[index])
  # interpolation
  spa_result_inter = 0
  spa_weight = 0
  spa_cu_dataset = fileDatas[index]
  spa_cu_before_dataset = fileDatas[index - 1]
  spa_cu_after_dataset = fileDatas[index + 1]
  spa_winSize_unilateral = 5 # n*2 + 1

  sampleLength =  6  #len(fileDatas)
  tem_result_inter = 0
  tem_weight = 0
  tem_back_index = index + sampleLength
  tem_forward_index = index - sampleLength
  if index + sampleLength > len(fileDatas) : tem_back_index = len(fileDatas)
  if index - sampleLength < 0 : tem_forward_index = -1
  tem_winSize_unilateral = 2  # n*2 + 1

  X_final = []
  origin_value = []

  MQC_data = MQC_File[MQC_File["MQC_Score"][0,index - 1]]  # [column, row]
  MQC_Score = MQC_data[:]
  MQC_data_before = MQC_File[MQC_File["MQC_Score"][0,index - 2]]  # [column, row]
  MQC_Score_before = MQC_data_before[:]
  MQC_data_after = MQC_File[MQC_File["MQC_Score"][0,index]]  # [column, row]
  MQC_Score_after = MQC_data_after[:]
  for pos in position_nan:    
    # print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    lc_type = LC_Value[pos[0]][pos[1]]    
    MQC_value = MQC_Score[pos[0]][pos[1]]
    or_value = spa_cu_dataset[pos[0]][pos[1]]
    or_before_value = spa_cu_before_dataset[pos[0]][pos[1]]
    or_after_value = spa_cu_after_dataset[pos[0]][pos[1]]
    final = or_value
    if MQC_value >= 70 and or_value <= 70: 
      X_final.append(or_value)
      origin_value.append(or_value)
      print('great')
    elif MQC_value < 70: 
    # spatial relationship
      if(or_value <= 70):
        origin_value.append(or_value)
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
            if LC_Value[i][j] == lc_type:
              euclideanDis = math.sqrt(math.pow((pos[0] - i), 2) + math.pow((pos[1] - j), 2))
              # 在欧氏距离的基础上再按照MQC比重分配
              numerator[0] += (euclideanDis * spa_cu_before_dataset[i][j] * MQC_Score_before[i][j] * 0.01)
              numerator[1] += (euclideanDis * spa_cu_dataset[i][j] * MQC_Score[i][j] * 0.01)
              numerator[2] += (euclideanDis * spa_cu_after_dataset[i][j] * MQC_Score_after[i][j] * 0.01)
              denominator[0] += euclideanDis * MQC_Score_before[i][j] * 0.01
              denominator[1] += euclideanDis * MQC_Score[i][j] * 0.01
              denominator[2] += euclideanDis * MQC_Score_after[i][j] * 0.01
        # 当5*5范围内无相同lc时，当前空间相关性计算值和权重均为0 （此方案不行）
        if denominator[0] > 0:
          spa_result_inter = round(numerator[1]/denominator[1])
          before_weight = abs(round(numerator[0]/denominator[0]) - or_before_value)
          after_weight = abs(round(numerator[2]/denominator[2]) - or_after_value)
          spa_weight = before_weight + after_weight

    # temporal relationship
        weight = 10 
        numerator = [] # 分子
        denominator = []  # 分母
        tem_index = 0
        tem_wei_count = 0
        tem_row_before = 0 
        tem_row_after = 2400
        tem_col_before = 0 
        tem_col_after = 2400
        wei_len = 1

        if pos[0]- tem_winSize_unilateral > 0 : tem_row_before = pos[0]- tem_winSize_unilateral
        if pos[0] + tem_winSize_unilateral < 2400 : tem_row_after = pos[0] + tem_winSize_unilateral
        if pos[1]- tem_winSize_unilateral > 0 : tem_col_before = pos[1]- tem_winSize_unilateral
        if pos[1] + tem_winSize_unilateral < 2400 : tem_col_after = pos[1] + tem_winSize_unilateral
        # print(tem_row_before, tem_row_after, tem_col_before, tem_col_after)
        for i in range(tem_row_before, tem_row_after):
          for j in range(tem_col_before, tem_col_after):
            if LC_Value[i][j] == lc_type:
              forward_index = index - 1
              backward_index = index + 1
              forward_count = 0
              backward_count = 0
              numerator.append(0)
              denominator.append(0)
              while (forward_index > tem_forward_index):
                value = fileDatas[forward_index][i][j]
                if(value <= 70):
                  # MQC_data = MQC_File[MQC_File["MQC_Score"][0,forward_index]]  # [column, row]
                  # MQC_Score = MQC_data[:]
                  # numerator[tem_index] += value * (weight - forward_count) * MQC_Score[i][j] * 0.01
                  # denominator[tem_index] += (weight - forward_count) * MQC_Score[i][j] * 0.01
                  numerator[tem_index] += value * (weight - forward_count)
                  denominator[tem_index] += (weight - forward_count)
                forward_index -= 1
                forward_count += wei_len
              while (backward_index < tem_back_index):
                value = fileDatas[backward_index][i][j]
                if(value <= 70):
                  # MQC_data = MQC_File[MQC_File["MQC_Score"][0,backward_index]]
                  # MQC_Score = MQC_data[:]
                  # numerator[tem_index] += value * (weight - backward_count) * MQC_Score[i][j] * 0.01
                  # denominator[tem_index] += (weight - backward_count) * MQC_Score[i][j] * 0.01
                  numerator[tem_index] += value * (weight - backward_count)
                  denominator[tem_index] += (weight - backward_count) 
                backward_index += 1
                backward_count += wei_len 
              if denominator[tem_index] != 0 :        
                inter = round(numerator[tem_index] / denominator[tem_index])
                if(i == pos[0] and j == pos[1]):
                  tem_result_inter = inter
                else:
                  dif_value = abs(inter - spa_cu_dataset[i][j])
                  tem_wei_count += dif_value
              # else : inter = 250       
              # if(i == pos[0] and j == pos[1]):
              #   tem_result_inter = inter
              # else:
              #   dif_value = abs(inter - spa_cu_dataset[i][j])
              #   tem_wei_count += dif_value
              tem_index += 1
        tem_weight = tem_wei_count
        # print(spa_weight, tem_weight)
        
    # total Calculation
        if (MQC_value >= 30 and MQC_value < 70):        
          final = round((or_value * MQC_value + spa_result_inter * spa_weight + tem_result_inter * tem_weight) / (MQC_value + spa_weight + tem_weight))  
          X_final.append(final)
        else :
          if spa_weight != 0 and tem_weight != 0 : 
            final = round((spa_result_inter * spa_weight + tem_result_inter * tem_weight) / (spa_weight + tem_weight))  
            X_final.append(final)
      
      # print(spa_result_inter, tem_result_inter)
    LAI_Result[pos[0]][pos[1]] = final


  colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
  bounds = [0,10,20,30,40,50,60,70,250]
  cmap = pltcolor.ListedColormap(colors)
  norm = pltcolor.BoundaryNorm(bounds, cmap.N)
  # plt.title('Filling Image', family='Times New Roman', fontsize=18)
  plt.imshow(LAI_Result, cmap=cmap, norm=norm)
  # cbar = plt.colorbar()
  # cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
# # 去除图片空白区域
  # fig = plt.gcf()
  # fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
  # plt.gca().xaxis.set_major_locator(plt.NullLocator())
  # plt.gca().yaxis.set_major_locator(plt.NullLocator())
  # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
  # plt.margins(0,0)
  # fig.savefig("./result/h27v07_png/h27v07_" + str(index + 1) + ".png", format='png', transparent=True, dpi=300, pad_inches = 0)
  # plt.show()

  # np.savetxt('./result/h27v07_data/h27v07_' + str(index + 1) ,LAI_Result)

# line chart
  # print(len(X_final), len(origin_value))
  # aa = np.arange(len(X_final))
  # plt.figure(figsize=(25, 6)) #宽，高
  # # plt.xlabel('Year', fontsize='14') 
  # plt.ylabel('Number', fontsize=15, family='Times New Roman')

  # line1=plt.plot(aa,origin_value, label='count', color='#fd7400',  marker='o', markersize=5)
  # line4=plt.plot(aa,X_final, label='count',color='#ffe117',  marker='o', markersize=5)
  # plt.legend(
  #   (line1[0],  line4[0]), 
  #   ('Origin',  'Final'),
  #   loc = 4, prop={'size':15, 'family':'Times New Roman'},
  #   )
  # plt.show()
  print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# part png
  # origin_part = []
  # result_part = []
  # for i in range(0, 500):
  #   origin_row = []
  #   result_row = []
  #   for j in range(1700, 2200):
  #     origin_row.append(fileDatas[index][i][j])
  #     result_row.append(LAI_Result[i][j])
  #   result_part.append(result_row)
  #   origin_part.append(origin_row)

  # colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
  # bounds = [0,10,20,30,40,50,60,70,250]
  # cmap = pltcolor.ListedColormap(colors)
  # norm = pltcolor.BoundaryNorm(bounds, cmap.N)
  # plt.title('Origin_part', family='Times New Roman', fontsize=18)
  # plt.imshow(origin_part, cmap=cmap, norm=norm)
  # cbar = plt.colorbar()
  # cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
  # # plt.savefig("./h27v07_test_pic/Origin_part.png", dpi=300)
  # plt.show()


  # plt.title('Filling_part', family='Times New Roman', fontsize=18)
  # plt.imshow(result_part, cmap=cmap, norm=norm)
  # cbar = plt.colorbar()
  # cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
  # # plt.savefig("./h27v07_test_pic/Filling_part.png", dpi=300)
  # plt.show()