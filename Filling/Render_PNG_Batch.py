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

# 绘制不同植被类型的LAI曲线（标准、添加误差的、提升后的）
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

# 计算整个tile的LAI差
def Average_LAI():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    Tem_improvedArray = []
    Spa_improvedArray = []
    # for i in range(3, 4):
    #     tem_data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
    #     spa_data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
    #     Tem_improvedArray.append(tem_data)
    #     Spa_improvedArray.append(spa_data)
    i = 33
    tem_data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
    spa_data = np.loadtxt('./Daily_cache/0518/Spa_LAI/LAI_%s' % i)
    Tem_improvedLAI = np.array(tem_data)
    Spa_improvedLAI = np.array(spa_data)


    render_LAI(StandLAI[i], title='LAI', issave=True, savepath='./Daily_cache/0518/Standard_LAI')
    render_LAI(InaccurateLAI[i], title='LAI', issave=True, savepath='./Daily_cache/0518/Inaccurate_LAI')
    render_LAI(Tem_improvedLAI, title='LAI', issave=True, savepath='./Daily_cache/0518/Impro_Tem_LAI')
    render_LAI(Spa_improvedLAI, title='LAI', issave=True, savepath='./Daily_cache/0518/Impro_Spa_LAI')

    # 46期均值相对标准数据的绝对差
    render_Img((StandLAI[i]-InaccurateLAI[i])/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Inaccurate')
    render_Img((StandLAI[i]-Tem_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Tem')
    render_Img((StandLAI[i]-Spa_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Spa')
    
    print(np.mean(np.abs(StandLAI[i]-InaccurateLAI[i])/10), np.mean(np.abs(StandLAI[i]-Tem_improvedLAI)/10), np.mean(np.abs(StandLAI[i]-Spa_improvedLAI)/10))
    
    # 当空间范围内无相同LC时，空间计算值为0
    # LandCover = np.load('../Simulation/Simulation_Dataset/LandCover.npy')
    # Mean_diff_Spa = mean_standard - mean_spa
    # print(np.nonzero(Mean_diff_Spa > 10))
    # pos = (62,494)
    # print(StandLAI[i, pos[0], pos[1]])
    # print(InaccurateLAI[i, pos[0], pos[1]])
    # print(Tem_improvedLAI[pos[0], pos[1]])
    # print(Spa_improvedLAI[pos[0], pos[1]])
    # print(Tem_improvedLAI[:, pos[0], pos[1]])
    # print(Spa_improvedLAI[:, pos[0], pos[1]])
    # print(LandCover[pos])
    