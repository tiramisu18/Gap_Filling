# 对比不同空间窗口大小下的W2（空间权重）的值

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


def Spatial_W2 (fileDatas, index, position_nan, LC_Value, MQC_File):
  print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


  # interpolation
  spa_result_inter = 0
  spa_weight = 0
  spa_cu_dataset = fileDatas[index]
  spa_cu_before_dataset = fileDatas[index - 1]
  spa_cu_after_dataset = fileDatas[index + 1]

  spa_winSize_unilateral = 10 # n*2 + 1



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

    if MQC_value < 70: 
    # spatial relationship
      if(or_value <= 70):
          
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
        # 当n*n范围内无相同lc时，当前空间相关性计算值和权重均为0 （此方案不行）
        if denominator[0] > 0:
          spa_result_inter = round(numerator[1]/denominator[1])
          before_weight = abs(round(numerator[0]/denominator[0]) - or_before_value)
          after_weight = abs(round(numerator[2]/denominator[2]) - or_after_value)
          spa_weight = before_weight + after_weight
    
    print(spa_weight)




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

