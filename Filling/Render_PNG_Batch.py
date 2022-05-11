import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy.ma as ma
import Draw_PoltLine

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

def render_Img (data, title='Algo Path', issave=False, savepath=''):
    plt.imshow(data, cmap = plt.cm.seismic)  # cmap= plt.cm.jet
    # plt.imshow(data, cmap = plt.cm.coolwarm) 
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

# LAI_Simu_Err = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70).npy')
# for i in range(0, 46):
#     render_LAI(LAI_Simu_Err[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s' % (i+1))
#     # render_LAI(LAI_Simu_Err[i], '2018_%d_Err'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s_addErr' % (i+1))


# Err_value = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Err_value.npy')
# Err_peren = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/Err_peren.npy')
# for i in range(3,4):
#     render_Img(Err_value[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/2018_%s_errValue' % (i+1))
#     # render_Img(Err_peren[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/2018_%s_errperen' % (i+1))
#     # render_LAI(LAI_Simu_Err[i], '2018_%d_Err'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s_addErr' % (i+1))

# 填补后的数据
# for i in range(2,3):
#     data_array = np.loadtxt('../Simulation/Filling/2018_%s' % i)
#     render_LAI(data_array, '2018_%d_Filling'% (1 + (i-1)*8),True, './Daily_cache/0316/Step2_LAI_Filling/2018_%s' % (i))

# 计算数据提升之后整个tile的46期与原始含有误差数据的RMSE
def calRMSE_Spa():
    improvedArray = []
    for i in range(1, 47):
        data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
        improvedArray.append(data)
    improvedLAI = np.array(improvedArray)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    calRMSE = np.sqrt((1/len(improvedArray))* np.sum(np.square(standLAI - improvedLAI), axis=0)) / 10
    print(np.mean(calRMSE))
    render_Img(calRMSE, title='RMSE', savepath='./Daily_cache/0506/RMSE_Spa_Improved', issave=False)
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    calRMSE_err = np.sqrt((1/len(improvedArray))* np.sum(np.square(standLAI - LAI_Simu_addErr), axis=0)) / 10
    print(np.mean(calRMSE_err))
    render_Img(calRMSE_err, title='RMSE', savepath='./Daily_cache/0506/RMSE', issave=False)

def calRMSE_Tem():
    improvedArray = []
    for i in range(1, 47):
        data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
        improvedArray.append(data)
    improvedLAI = np.array(improvedArray)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    calRMSE = np.sqrt((1/len(improvedArray))* np.sum(np.square(standLAI - improvedLAI), axis=0)) / 10
    print(np.mean(calRMSE))
    render_Img(calRMSE, title='RMSE', savepath='./Daily_cache/0506/RMSE_Tem_Improved', issave=False)
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    calRMSE_err = np.sqrt((1/len(improvedArray))* np.sum(np.square(standLAI - LAI_Simu_addErr), axis=0)) / 10    
    render_Img(calRMSE_err, title='RMSE', savepath='./Daily_cache/0506/RMSE', issave=True)

# 计算数据提升之后不同植被类型与原始含有误差数据的曲线变化
def landCover_Improved_Spa():    
    improvedArray = []
    for i in range(1, 47):
        data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
        improvedArray.append(data)
    improvedLAI = np.array(improvedArray)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    
    lc_type = 6
    aa = []
    bb = []
    cc = []
    for i in range(0, 46):
        improved_ma = ma.array((ma.masked_greater(improvedLAI[i], 70)), mask=(LandCover != lc_type))
        stand_ma = ma.array((ma.masked_greater(standLAI[i], 70)), mask=(LandCover != lc_type))
        addErr_ma = ma.array((ma.masked_greater(LAI_Simu_addErr[i], 70)), mask=(LandCover != lc_type))
        aa.append(np.mean(improved_ma)/10)
        bb.append(np.mean(stand_ma)/10)
        cc.append(np.mean(addErr_ma)/10)

    Draw_PoltLine.draw_polt_Line(np.arange(0, 361, 8),{
        'title': 'B%s'% lc_type ,
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [bb, cc, aa],
        'le_name': ['Standard','Inaccurate', 'Improved'],
        'color': ['#bfdb39', 'gray', '#fd7400'],
        'marker': ['o', ',', '^', '.' ],
        'size': False,
        'lineStyle': ['solid', 'dashed']
        },'./Daily_cache/0506/Spa_LC/lc_type_%s'% lc_type, True, 2)


def landCover_Improved_Tem():    
    improvedArray = []
    for i in range(1, 47):
        data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
        improvedArray.append(data)
    improvedLAI = np.array(improvedArray)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    
    lc_type = 1
    aa = []
    bb = []
    cc = []
    for i in range(0, 46):
        improved_ma = ma.array((ma.masked_greater(improvedLAI[i], 70)), mask=(LandCover != lc_type))
        stand_ma = ma.array((ma.masked_greater(standLAI[i], 70)), mask=(LandCover != lc_type))
        addErr_ma = ma.array((ma.masked_greater(LAI_Simu_addErr[i], 70)), mask=(LandCover != lc_type))
        aa.append(np.mean(improved_ma)/10)
        bb.append(np.mean(stand_ma)/10)
        cc.append(np.mean(addErr_ma)/10)

    Draw_PoltLine.draw_polt_Line(np.arange(0, 361, 8),{
        'title': 'B%s'% lc_type ,
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [bb, cc, aa],
        'le_name': ['Standard','Inaccurate', 'Improved'],
        'color': ['#bfdb39', 'gray', '#fd7400'],
        'marker': ['o', ',', '^', '.' ],
        'size': False,
        'lineStyle': ['solid', 'dashed']
        },'./Daily_cache/0506/Tem_LC/lc_type_%s'% lc_type, True, 2)

def Average_LAI():
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    Tem_improvedArray = []
    Spa_improvedArray = []
    for i in range(1, 47):
        tem_data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
        spa_data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
        Tem_improvedArray.append(tem_data)
        Spa_improvedArray.append(spa_data)
    Tem_improvedLAI = np.array(Tem_improvedArray)
    Spa_improvedLAI = np.array(Spa_improvedArray)

    mean_standard = np.mean(standLAI, axis=0)
    mean_inaccurate = np.mean(LAI_Simu_addErr, axis=0)
    mean_tem = np.mean(Tem_improvedLAI, axis=0)
    mean_spa = np.mean(Spa_improvedLAI, axis=0)
    # 46期均值
    # render_LAI(mean_standard, title='LAI', issave=True, savepath='./Daily_cache/0506/Mean_Standard_LAI')
    # render_LAI(mean_inaccurate, title='LAI', issave=True, savepath='./Daily_cache/0506/Mean_Inaccurate_LAI')
    # render_LAI(mean_tem, title='LAI', issave=True, savepath='./Daily_cache/0506/Mean_Impro_Tem_LAI')
    # render_LAI(mean_spa, title='LAI', issave=True, savepath='./Daily_cache/0506/Mean_Impro_Spa_LAI')

    # 46期均值相对标准数据的绝对差
    render_Img((mean_standard-mean_inaccurate)/10,title='', issave=True, savepath='./Daily_cache/0506/Mean_diff_Inaccurate')
    render_Img((mean_standard-mean_tem)/10,title='', issave=True, savepath='./Daily_cache/0506/Mean_diff_Tem')
    render_Img((mean_standard-mean_spa)/10,title='', issave=True, savepath='./Daily_cache/0506/Mean_diff_Spa')
    
    print(np.mean(np.abs(mean_standard-mean_inaccurate)/10), np.mean(np.abs(mean_standard-mean_tem)/10), np.mean(np.abs(mean_standard-mean_spa)/10))
    
    # 当空间范围内无相同LC时，空间计算值为0
    # LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    # Mean_diff_Spa = mean_standard - mean_spa
    # print(np.nonzero(Mean_diff_Spa > 10))
    # pos = (62,494)
    # print(mean_standard[pos], mean_spa[pos])
    # print(standLAI[:, pos[0], pos[1]])
    # print(LAI_Simu_addErr[:, pos[0], pos[1]])
    # print(Tem_improvedLAI[:, pos[0], pos[1]])
    # print(Spa_improvedLAI[:, pos[0], pos[1]])
    # print(LandCover[pos])
    