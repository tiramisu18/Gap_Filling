import numpy as np
import numpy.ma as ma
import math
import time
import Public_Methods
import Newtons_Method
import sympy as sp


# 整个tile计算
def Temporal_Cal_Matrix_Tile (fileDatas, index, landCover, qualityControl, half_temLength, ses_pow, position=(0,0)):
    # calculate smoothing parameter (half temLength)
    paraRightHalf = []
    for i in range(0, half_temLength):
        para = round(ses_pow * (1 - ses_pow) ** i, 4)
        paraRightHalf.append(para)

    back_count = len(fileDatas) - index - 1 if index + half_temLength > len(fileDatas) - 1  else half_temLength
    forward_count = index if index - half_temLength < 0  else half_temLength
    paraLeftHalf = paraRightHalf[:forward_count]
    paraLeftHalf.reverse()
    smoothingList = paraLeftHalf + paraRightHalf[:back_count]
    smoothingArray = np.array(smoothingList)

    # 将lai值大于70的位置mask，但mask后影响此处的计算，因此暂时将其填充为0
    LAIDatas = ma.filled(ma.masked_greater(np.delete(fileDatas[index - forward_count:index + back_count + 1, ...], forward_count, 0), 70), fill_value=0)
    # LAIDatas = np.delete(fileDatas[index - forward_count:index + back_count + 1, ...], forward_count, 0) 
    QCDatas = np.delete(qualityControl[index - forward_count:index + back_count + 1, ...],forward_count, 0) 

    SPara = smoothingArray.reshape(len(smoothingList),1,1)
    numerators = (LAIDatas * QCDatas * SPara).sum(axis=0)
    denominators = (QCDatas * SPara).sum(axis=0)
    LAIImprovedDatas = np.round(numerators / denominators)
    # 目前，255填充值通过计算修补了部分数据，下面两步会将原来的填充值255还原
    pos = fileDatas[index, ...].__gt__(70)
    LAIImprovedDatas[pos] = fileDatas[index, ...][pos]

    # u, count = np.unique(fileDatas[index , ...], return_counts=True)  
    # intersect = (LAIRange <= 70) == (lcRange == posLC)
    # filter = np.nonzero(intersect == True) #get the indices of elements that satisfy the conditions, return array (row indices, column indices)  
    return LAIImprovedDatas

# 整个tile计算
def Spatial_Cal_Matrix_Tile(fileDatas, index, landCover, qualityControl, euc_pow, half_winWidth, position=(15,15)):
    # print('begin_tem_v1', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    LAIImprovedDatas = np.array(fileDatas[index, ...]).copy()
    rawLAI = ma.masked_greater(fileDatas[index, ...], 70) # 因为存在lc分类错误的情况（植被类型下的lai值为254【水域】）所以先将lai值大于70的位置mask
    rawQC = ma.array(qualityControl[index, ...], mask=rawLAI.mask) 
    rowSize = rawLAI.shape[0]
    colSize = rawLAI.shape[1]
    for lcType in range(1, 9):
        # lcma = ma.masked_not_equal(landCover, lcType) 
        # rawLAIMasked = ma.array(rawLAI, mask=lcma.mask) # 再按照不同的lc类型进行mask
        # rawQCMasked = ma.array(rawQC, mask=lcma.mask) 
        rawLAIMasked = ma.array(rawLAI, mask=landCover.__ne__(lcType)) # 再按照不同的lc类型进行mask
        rawQCMasked = ma.array(rawQC, mask=landCover.__ne__(lcType))
        EdLAIList = []
        EdQCList = []
        EdList = []   
        for i in range(-half_winWidth, half_winWidth+1):
            for j in range(-half_winWidth, half_winWidth+1):
                mm = ma.masked_all(rawLAI.shape)
                nn = ma.masked_all(rawLAI.shape)           
                if i == 0 and j == 0: continue
                if i <= 0 :
                    if j <= 0: 
                        mm[abs(i):, abs(j):] = rawLAIMasked[:rowSize-abs(i), :colSize-abs(j)]
                        nn[abs(i):, abs(j):] = rawQCMasked[:rowSize-abs(i), :colSize-abs(j)]
                    else: 
                        mm[abs(i):, 0:colSize-j] = rawLAIMasked[:rowSize-abs(i), j:]
                        nn[abs(i):, 0:colSize-j] = rawQCMasked[:rowSize-abs(i), j:]
                else:
                    if j <= 0: 
                        mm[0:rowSize-i, abs(j):] = rawLAIMasked[i:, :colSize-abs(j)]
                        nn[0:rowSize-i, abs(j):] = rawQCMasked[i:, :colSize-abs(j)]
                    else: 
                        mm[0:rowSize-i, 0:colSize-j] = rawLAIMasked[i:, j:]
                        nn[0:rowSize-i, 0:colSize-j] = rawQCMasked[i:, j:]
                EdLAIList.append(mm)
                EdQCList.append(nn)
                EdList.append((math.sqrt(abs(i) ** 2 + abs(j) ** 2) ** -euc_pow))
        EdLAIArray = ma.array(EdLAIList)
        # EdLAIArray = ma.stack(tuple(EdLAIList), axis=0) # 将EdLAIList列表转为元组，按照默认最外面的轴堆叠元组
        EdQCArray = ma.array(EdQCList)
        EdArray = np.array(EdList).reshape(-1, 1, 1)
        numerators = (EdLAIArray * EdQCArray * EdArray).sum(axis=0)
        denominators = (EdArray * EdQCArray).sum(axis=0)
        LAIImprovedData = ma.round(numerators / denominators)
        pos = landCover.__eq__(lcType)
        # point = (62,494) 
        # 找出范围内无相同LC的像元，赋为原始值
        no_lc = np.logical_and(pos, LAIImprovedData.mask)
        LAIImprovedData[no_lc] = fileDatas[index, ...][no_lc]
        LAIImprovedDatas[pos] = LAIImprovedData[pos]
        
    # print('Tile_Spa', LAIImprovedDatas[position])
    # 目前，255填充值通过计算修补了部分数据，下面两步会将原来的填充值255还原
    pos1 = fileDatas[index, ...].__gt__(70)
    LAIImprovedDatas[pos1] = fileDatas[index, ...][pos1]
    return LAIImprovedDatas

# 整个tile权重计算
def Calculate_Weight(TemLAI, SpaLAI, RawLAI, LandCover, qualityControl, pos):  
    winSize = 4  
    # print('Tem',TemLAI[pos[0]-size:pos[0]+size+1, pos[1]-size:pos[1]+size+1])
    # print('LC',LandCover[pos[0]-size:pos[0]+size+1, pos[1]-size:pos[1]+size+1])
    # print('QC',qualityControl[pos[0]-size:pos[0]+size+1, pos[1]-size:pos[1]+size+1])
    rowSize = RawLAI.shape[0]
    colSize = RawLAI.shape[1]
    for lcType in range(1, 2):
        temMask = ma.array(ma.array(TemLAI, mask=LandCover != lcType), mask=qualityControl < 8)
        spaMask = ma.array(ma.array(SpaLAI, mask=LandCover != lcType), mask=qualityControl < 8)
        rawMask = ma.array(ma.array(RawLAI, mask=LandCover != lcType), mask=qualityControl < 8)

        temList = []
        spaList = []
        rawList = []
        # EdLAIArray = ma.array([])    
        for i in range(-winSize, winSize+1):
            for j in range(-winSize, winSize+1):
                mm = ma.masked_all(RawLAI.shape)
                nn = ma.masked_all(RawLAI.shape) 
                zz = ma.masked_all(RawLAI.shape)
                if i == 0 and j == 0: continue
                if i <= 0 :
                    if j <= 0: 
                        mm[abs(i):, abs(j):] = temMask[:rowSize-abs(i), :colSize-abs(j)]
                        nn[abs(i):, abs(j):] = spaMask[:rowSize-abs(i), :colSize-abs(j)]
                        zz[abs(i):, abs(j):] = rawMask[:rowSize-abs(i), :colSize-abs(j)]
                    else: 
                        mm[abs(i):, 0:colSize-j] = temMask[:rowSize-abs(i), j:]
                        nn[abs(i):, 0:colSize-j] = spaMask[:rowSize-abs(i), j:]
                        zz[abs(i):, 0:colSize-j] = rawMask[:rowSize-abs(i), j:]
                else:
                    if j <= 0: 
                        mm[0:rowSize-i, abs(j):] = temMask[i:, :colSize-abs(j)]
                        nn[0:rowSize-i, abs(j):] = spaMask[i:, :colSize-abs(j)]
                        zz[0:rowSize-i, abs(j):] = rawMask[i:, :colSize-abs(j)]
                    else: 
                        mm[0:rowSize-i, 0:colSize-j] = temMask[i:, j:]
                        nn[0:rowSize-i, 0:colSize-j] = spaMask[i:, j:]
                        zz[0:rowSize-i, 0:colSize-j] = rawMask[i:, j:]
                temList.append(mm)
                spaList.append(nn)
                rawList.append(zz)
        temArray = ma.array(temList)
        spaArray = ma.array(spaList)
        rawArray = ma.array(rawList)
        print('2', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        weight = Structure_Lagrange(temArray, spaArray, rawArray)
        # print(mse[:,0])
        # temWeight = []
        # spaWeight = []
        # for row in mse:
        #     temRow = []
        #     spaRow = []
        #     for ele in row:
        #         print(ele)
        #         print(ele[0])
        #         print(ele[0].keys())
        #         print(dict(ele[0]).get('x', default=None))
                # spaRow.append(ele[0]['X'])
        #         temRow.append(ele[0]['y'])
        #     temWeight.append(temRow)
        #     spaWeight.append(spaRow)
        # print(temWeight)
        # print(np.array(temWeight).shape)
        # print(mse.shape, mse[pos[0],pos[1]])
        # aa = temArray[:,pos[0],pos[1]]
        # print(aa[~aa.mask])
        # print(spaArray[:,pos[0],pos[1]])
        # print(rawArray[:,pos[0],pos[1]])
        # numerators = (EdLAIArray * EdQCArray * EdArray).sum(axis=0)
        # denominators = (EdArray * EdQCArray).sum(axis=0)
        # LAIImprovedData = ma.round(numerators / denominators)
        # pos = landCover.__eq__(lcType)
        # LAIImprovedDatas[pos] = LAIImprovedData[pos]   
        
    # 单个点    
    # lcType = LandCover[pos]
    # print(lcType)
    # temMask = ma.array(ma.array(TemLAI, mask=LandCover != lcType), mask=qualityControl < 8)
    # spaMask = ma.array(ma.array(SpaLAI, mask=LandCover != lcType), mask=qualityControl < 8)
    # rawMask = ma.array(ma.array(RawLAI, mask=LandCover != lcType), mask=qualityControl < 8)

    # partTemMask = temMask[pos[0]-winSize:pos[0]+winSize+1, pos[1]-winSize:pos[1]+winSize+1]
    # partSpaMask = spaMask[pos[0]-winSize:pos[0]+winSize+1, pos[1]-winSize:pos[1]+winSize+1]
    # partRawMask = rawMask[pos[0]-winSize:pos[0]+winSize+1, pos[1]-winSize:pos[1]+winSize+1]

    # partTem = partTemMask[~partTemMask.mask]
    # partSpa = partSpaMask[~partSpaMask.mask]
    # partRaw = np.round(partRawMask[~partRawMask.mask],1)
    # print(partTem.tolist())
    # print(partSpa.tolist())
    # print(partRaw.tolist())
    # x, y, k= sp.symbols('x y k')
    # dataSum1 = 0
    # for i in range(0, len(partTem)):
    #     final = (partSpa[i] * x + partTem[i] * y) / (x + y)
    #     dataSum1 = dataSum1 + ((final - partRaw[i])**2)
    # # print(dataSum1)
    return 

# 构造函数
def Structure_Lagrange(Tem, Spa, Raw):
    x, y= sp.symbols('x y', real=True)
    Spa = np.array(ma.filled(Spa, 0))
    Tem = np.array(ma.filled(Tem, 0))
    Raw = np.array(ma.filled(Raw, 0))
    final = (Spa * x + Tem * y) / (x + y)
    # mse = (np.square(final - Raw)).sum(axis=0)
    mse = np.mean(np.square(final - Raw), axis=0)
    weight = Newtons_Method.simulated_martix(mse, x, y)
    return weight


# 单像元矩阵计算
def Temporal_Cal_Matrix_Pixel (fileDatas, index, position, landCover, qualityControl, temporalLength, winSize, SES_pow):
    # print('begin_tem', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    paraRightHalf = []
    for i in range(0, temporalLength):
        para = round(SES_pow * (1 - SES_pow) ** i, 4)
        paraRightHalf.append(para)

    back_count = len(fileDatas) - index - 1 if index + temporalLength > len(fileDatas) - 1  else 6
    forward_count = index if index - temporalLength < 0  else 6
    paraLeftHalf = paraRightHalf[:forward_count]
    paraLeftHalf.reverse()
    smoothingList = paraLeftHalf + paraRightHalf[:back_count]
    smoothingArray = np.array(smoothingList)

    targetLAI = fileDatas[index, position]
    targetLAIList = []
    targetQCList = []
    for i in range(index - forward_count, index + back_count + 1):
        if i != index:
            targetLAIList.append(fileDatas[i, position])
            targetQCList.append(qualityControl[i, position])
    targetLAIArray = np.array(targetLAIList)
    targetQCArray = np.array(targetQCList)
    numerator = targetLAIArray * smoothingArray * targetQCArray
    denominator = smoothingArray * targetQCArray   
    improvedValue = numerator.sum() / denominator.sum()   

    pos = position
    row_before = pos[0]- winSize if pos[0]- winSize > 0 else 0
    row_after = pos[0] + winSize + 1 if pos[0] + winSize < len(landCover[0]) else len(landCover[0])
    col_before = pos[1]- winSize if pos[1]- winSize > 0 else 0 
    col_after = pos[1] + winSize + 1 if pos[1] + winSize < len(landCover) else len(landCover)

    posLC = landCover[pos[0]][pos[1]] 
    LAIRange = fileDatas[index, row_before:row_after, col_before:col_after]      
    lcRange = landCover[row_before:row_after, col_before:col_after]
    intersect = (LAIRange <= 70) == (lcRange == posLC)
    filter = np.nonzero(intersect == True) #get the indices of elements that satisfy the conditions, return array (row indices, column indices)
    list_of_coordinates = list(zip(filter[0], filter[1])) #generate a list of coordinates
    rawLAITemList = []
    QCList = []
    rawLAIList = []
    list_of_coordinates.remove((pos[0] - row_before, pos[1] - col_before))
    # print('length', len(list_of_coordinates))
    for coord in list_of_coordinates:
        rawCoordRow = coord[0] + row_before
        rawCoordCol = coord[1] + col_before
        rawValueOne = []
        QCOne = []
        rawLAIList.append(fileDatas[index, rawCoordRow, rawCoordCol])
        for i in range(index - forward_count, index + back_count + 1):
            if i != index:
                rawValueOne.append(fileDatas[i, rawCoordRow, rawCoordCol])
                QCOne.append(qualityControl[i, rawCoordRow, rawCoordCol])
        rawLAITemList.append(rawValueOne)
        QCList.append(QCOne)

    # print(np.array(rawLAITemList).shape)
    rawLAITemArray = np.array(rawLAITemList)
    QCArray = np.array(QCList)
    rawLAIArray = np.array(rawLAIList)
    numerators = rawLAITemArray * QCArray * smoothingArray
    denominators = QCArray * smoothingArray 
    improvedValues = abs((numerators.sum(axis=1) / denominators.sum(axis=1)) - rawLAIArray)
    weight = improvedValues.sum()/len(improvedValues)
    print('m1', improvedValue, weight, targetLAI)
    return {'weight': weight, 'filling': improvedValue, 'or_value': targetLAI}


# 逐像元循环计算（哒咩）
def Temporal_Cal (fileDatas, index, Filling_Pos, LC_info, QC_File, temporalLength, tem_winSize_unilateral, SES_pow):
    # print('begin_tem_previous', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  # interpolation
    cu_dataset = fileDatas[index]
    # temporalLength =  6  
    tem_filling_value = 0
    tem_weight = 0
    tem_back_index = index + temporalLength
    tem_forward_index = index - temporalLength
    if index + temporalLength > len(fileDatas) - 1 : tem_back_index = len(fileDatas) - 1
    if index - temporalLength < 0 : tem_forward_index = 0
    # tem_winSize_unilateral = 2  # n*2 + 1
     
    pos = Filling_Pos

    lc_type = LC_info[pos[0]][pos[1]] 
    or_value = cu_dataset[pos[0]][pos[1]]    
    numerator = [] # 分子
    denominator = []  # 分母
    tem_index = 0
    tem_wei_count = 0
    tem_row_before = 0 
    tem_row_after = len(LC_info[0])
    tem_col_before = 0 
    tem_col_after = len(LC_info)
    valid_lc = 0

    if pos[0]- tem_winSize_unilateral > 0 : tem_row_before = pos[0]- tem_winSize_unilateral
    if pos[0] + tem_winSize_unilateral < len(LC_info[0]) : tem_row_after = pos[0] + tem_winSize_unilateral + 1
    if pos[1]- tem_winSize_unilateral > 0 : tem_col_before = pos[1]- tem_winSize_unilateral
    if pos[1] + tem_winSize_unilateral < len(LC_info) : tem_col_after = pos[1] + tem_winSize_unilateral + 1
    for i in range(tem_row_before, tem_row_after):
        for j in range(tem_col_before, tem_col_after):
            if LC_info[i][j] == lc_type:
                forward_index = index - 1
                backward_index = index + 1
                forward_i = 1
                backward_i = 1
                numerator.append(0)
                denominator.append(0)
                while (forward_index >= tem_forward_index):
                    value = fileDatas[forward_index][i][j]                    
                    tem_SES = SES_pow * (1 - SES_pow)**(forward_i - 1)              
                    if(value <= 70):
                        QC_Score = QC_File[forward_index] 
                        numerator[tem_index] += value * tem_SES * QC_Score[i][j] 
                        denominator[tem_index] += tem_SES * QC_Score[i][j]                                
                    forward_index -= 1
                    forward_i += 1                    
                while (backward_index <= tem_back_index):
                    value = fileDatas[backward_index][i][j]
                    tem_SES = SES_pow * math.pow((1 - SES_pow), backward_i - 1)                                    
                    if(value <= 70):
                        QC_Score = QC_File[backward_index]
                        numerator[tem_index] += value * tem_SES * QC_Score[i][j]
                        denominator[tem_index] += tem_SES * QC_Score[i][j]  
                    backward_index += 1
                    backward_i += 1
                if denominator[tem_index] != 0 :                           
                    inter = numerator[tem_index] / denominator[tem_index]
                    if(i == pos[0] and j == pos[1]):                                              
                        tem_filling_value = round(inter)
                    else :
                        valid_lc +=1 
                        dif_value = abs(inter - cu_dataset[i][j])
                        tem_wei_count += dif_value
                # else: 
                #     tem_filling_value = or_value
                #     print('eq 0 ', i, valid_lc, tem_filling_value)
                tem_index += 1
            # print('winsiz', i, j, tem_row_before, tem_row_after, tem_col_before, tem_col_after)        
    if valid_lc == 0 :
        tem_weight = 0 
        print('Tem eq zero', tem_winSize_unilateral, pos)
    else :
        tem_weight = (round(tem_wei_count/valid_lc, 2)) 
    # print('end_tem', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('previous', tem_filling_value, tem_weight, or_value)
    return {'weight': tem_weight, 'filling': tem_filling_value, 'or_value': or_value}

def Spatial_Cal (fileDatas, index, Filling_Pos, LC_info, QC_File, EUC_pow, spa_winSize_unilateral):
    # print('begin_spa', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    spa_filling_value = 0
    spa_weight = 0
    spa_cu_dataset = fileDatas[index]
    spa_cu_before_dataset = fileDatas[index - 1]
    spa_cu_after_dataset = fileDatas[index + 1]
    # spa_winSize_unilateral = 10 # n*2 + 1

    QC_Score = QC_File[index]
    QC_Score_before = QC_File[index - 1]
    QC_Score_after = QC_File[index + 1]
    pos = Filling_Pos  

    lc_type = LC_info[pos[0]][pos[1]]    
    or_before_value = spa_cu_before_dataset[pos[0]][pos[1]]
    or_after_value = spa_cu_after_dataset[pos[0]][pos[1]]
    or_value = spa_cu_dataset[pos[0]][pos[1]]      
    spa_row_before = 0
    spa_row_after = len(LC_info[0])
    spa_col_before = 0
    spa_col_after = len(LC_info)
    if pos[0] - spa_winSize_unilateral > 0 : spa_row_before = pos[0] - spa_winSize_unilateral
    if pos[0] + spa_winSize_unilateral < len(LC_info[0]) : spa_row_after = pos[0] + spa_winSize_unilateral + 1
    if pos[1] - spa_winSize_unilateral > 0 : spa_col_before = pos[1] - spa_winSize_unilateral
    if pos[1] + spa_winSize_unilateral < len(LC_info) : spa_col_after = pos[1] + spa_winSize_unilateral + 1
    euc = []  
    numerator = [0] * 3 # 分子
    denominator = [0] * 3  # 分母   
    for i in range(spa_row_before, spa_row_after):
        for j in range(spa_col_before, spa_col_after):
            if LC_info[i][j] == lc_type and spa_cu_dataset[i][j] <= 70:
                euclideanDis = math.sqrt(math.pow((pos[0] - i), 2) + math.pow((pos[1] - j), 2))
                if euclideanDis != 0 : euclideanDis = math.pow(euclideanDis, -EUC_pow)
                # 在欧氏距离的基础上再按照MQC比重分配
                numerator[0] += (euclideanDis * spa_cu_before_dataset[i][j] * QC_Score_before[i][j])
                numerator[1] += (euclideanDis * spa_cu_dataset[i][j] * QC_Score[i][j])
                numerator[2] += (euclideanDis * spa_cu_after_dataset[i][j] * QC_Score_after[i][j])
                denominator[0] += euclideanDis * QC_Score_before[i][j]
                denominator[1] += euclideanDis * QC_Score[i][j]
                denominator[2] += euclideanDis * QC_Score_after[i][j]
    #         if i == pos[0] and j == pos[1]: continue
    #         if LC_info[i][j] == lc_type:
    #             euc.append(math.pow(math.sqrt(math.pow((pos[0] - i), 2) + math.pow((pos[1] - j), 2)), -EUC_pow))
    #             lai.append(spa_cu_dataset[i][j])
    #             qc.append(QC_Score[i][j])
    # print('euc', euc, len(euc))  
    # print('lai', lai, len(lai))  
    # print('qc', qc, len(qc))  

    # 当n*n范围内无相同lc时，使用原始值填充
    if denominator[0] > 0 and denominator[1] > 0 and denominator[2] > 0: # 边界时会存在0 可改进
        # spa_filling_value = round(numerator[1]/denominator[1])
        spa_filling_value = numerator[1]/denominator[1]
        before_weight = abs((numerator[0]/denominator[0]) - or_before_value)
        after_weight = abs((numerator[2]/denominator[2]) - or_after_value)
        spa_weight = round((before_weight + after_weight) / 2, 2)
    else : 
        spa_filling_value = or_value
        spa_weight = 0
        print('Spa eq zero', spa_winSize_unilateral, pos)

    # print('end_spa', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print({'weight': spa_weight, 'filling': spa_filling_value, 'or_value': or_value})
    return {'weight': spa_weight, 'filling': spa_filling_value, 'or_value': or_value}

def Fill_Pixel (fileDatas, index, Filling_Pos, LC_info, QC_File, temporalLength, tem_winSize_unilateral, SES_pow, EUC_pow, spa_winSize_unilateral): 
    # LAI_Result = copy.deepcopy(fileDatas[index])
    # interpolation
    Or_value = []
    Fil_tem = []
    Fil_spa = []
    Fil_value = []
    Tem_W = []
    Spa_W = []
    Qc_W = []

    for pos in Filling_Pos:    
        tem_ob = Temporal_Cal (fileDatas, index, pos, LC_info, QC_File, temporalLength, tem_winSize_unilateral, SES_pow)
        spa_ob = Spatial_Cal (fileDatas, index, pos, LC_info, QC_File, EUC_pow, spa_winSize_unilateral) 
        QC_value = QC_File[index][pos[0]][pos[1]]  
        or_value = fileDatas[index][pos[0]][pos[1]] 
        final = or_value
        spa_filling_value = spa_ob['filling']
        spa_weight = spa_ob['weight']
        tem_filling_value = tem_ob['filling']
        tem_weight = tem_ob['weight']

        Or_value.append(or_value)
        Fil_tem.append(tem_filling_value)
        Fil_spa.append(spa_filling_value)
        Tem_W.append(tem_weight)
        Spa_W.append(spa_weight)
        Qc_W.append(QC_value)
        # total Calculation
        # final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight + or_value * QC_value) / (spa_weight + tem_weight + QC_value)) 
        # try: 
        #     if spa_weight != 0 and tem_weight != 0:
        #         final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight)) 
        # except:
        #     print(spa_filling_value, spa_weight, tem_filling_value, tem_weight)
        if (QC_value >= 8):        
            final = round((or_value * QC_value + spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (QC_value + spa_weight + tem_weight))  
        else :
            if spa_weight != 0 or tem_weight != 0 : 
                final = round((spa_filling_value * spa_weight + tem_filling_value * tem_weight) / (spa_weight + tem_weight))  
        
        Fil_value.append(final)

    # print({'tem': Fil_tem, 'spa': Fil_spa, 'Fil': Fil_value})      
    # LAI_Result[pos[0]][pos[1]] = final
    return {'Tem': Fil_tem, 'Spa': Fil_spa, 'Fil': Fil_value, 'Or': Or_value, 'T_W': Tem_W, 'S_W': Spa_W, 'Qc_W': Qc_W}
 
