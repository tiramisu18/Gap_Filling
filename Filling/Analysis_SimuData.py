import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy.ma as ma

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
    # plt.imshow(data, cmap = plt.cm.seismic)  # cmap= plt.cm.jet
    plt.imshow(data, cmap = plt.cm.coolwarm) 
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()


# 计算数据提升之后整个tile46期与原始含有误差数据的RMSE
def calRMSE_allTile():
    improvedArray_spa = []
    improvedArray_tem = []
    for i in range(1, 47):
        improvedArray_spa.append(np.loadtxt('./Daily_cache/0522/Spa_LAI/LAI_%s' % i))
        improvedArray_tem.append(np.loadtxt('./Daily_cache/0522/Tem_LAI/LAI_%s' % i))
    improvedLAI_spa = np.array(improvedArray_spa)
    improvedLAI_tem = np.array(improvedArray_tem)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    # Spatial
    calRMSE_spa = np.sqrt((1/len(improvedArray_spa))* np.sum(np.square(standLAI - improvedLAI_spa), axis=0)) / 10
    print(np.mean(calRMSE_spa))
    render_Img(calRMSE_spa, title='RMSE', savepath='./Daily_cache/0530/RMSE_Spa_Improved', issave=True)
    # Temporal
    calRMSE_tem = np.sqrt((1/len(improvedArray_tem))* np.sum(np.square(standLAI - improvedLAI_tem), axis=0)) / 10
    print(np.mean(calRMSE_tem))
    render_Img(calRMSE_tem, title='RMSE', savepath='./Daily_cache/0530/RMSE_Tem_Improved', issave=True)  
    # Inaccurate
    LAI_Simu_addErr = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    calRMSE_inacc = np.sqrt((1/len(improvedArray_spa))* np.sum(np.square(standLAI - LAI_Simu_addErr), axis=0)) / 10
    print(np.mean(calRMSE_inacc))
    render_Img(calRMSE_inacc, title='RMSE', savepath='./Daily_cache/0530/RMSE', issave=True)

# 计算单独一期整个tile的LAI差
def diff_LAI():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    Tem_improvedArray = []
    Spa_improvedArray = []
    # for i in range(3, 4):
    #     tem_data = np.loadtxt('./Daily_cache/0506/Tem_LAI/LAI_%s' % i)
    #     spa_data = np.loadtxt('./Daily_cache/0506/Spa_LAI/LAI_%s' % i)
    #     Tem_improvedArray.append(tem_data)
    #     Spa_improvedArray.append(spa_data)
    i = 23
    tem_data = np.loadtxt('./Daily_cache/0522/Tem_LAI/LAI_%s' % i)
    spa_data = np.loadtxt('./Daily_cache/0522/Spa_LAI/LAI_%s' % i)
    Tem_improvedLAI = np.array(tem_data)
    Spa_improvedLAI = np.array(spa_data)


    render_LAI(StandLAI[i], title='LAI', issave=True, savepath='./Daily_cache/0522/Standard_LAI')
    render_LAI(InaccurateLAI[i], title='LAI', issave=True, savepath='./Daily_cache/0522/Inaccurate_LAI')
    render_LAI(Tem_improvedLAI, title='LAI', issave=True, savepath='./Daily_cache/0522/Impro_Tem_LAI')
    render_LAI(Spa_improvedLAI, title='LAI', issave=True, savepath='./Daily_cache/0522/Impro_Spa_LAI')

    # 相对标准数据的绝对差
    Ina = (StandLAI[i]-InaccurateLAI[i])/10
    Tem = (StandLAI[i]-Tem_improvedLAI)/10
    Spa = (StandLAI[i]-Spa_improvedLAI)/10
    # render_Img((StandLAI[i]-InaccurateLAI[i])/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Inaccurate')
    # render_Img((StandLAI[i]-Tem_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Tem')
    # render_Img((StandLAI[i]-Spa_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Spa')
    data = [Ina, Tem, Spa]
    fig, axs = plt.subplots(1, 3)
    # fig.suptitle('Multiple images')
    images = []

    for j in range(3):
            # Generate data with a range that varies from one plot to the next.
            # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
        images.append(axs[j].imshow(data[j], cmap = plt.cm.seismic))
        axs[j].axis('off')
            # axs[i, j].width = 

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs,  fraction=.1)
    plt.savefig('./Daily_cache/0522/diff_%s'% i, dpi=300)
    plt.show()
    print(np.mean(np.abs(Ina)), np.mean(np.abs(Tem)), np.mean(np.abs(Spa)))
    
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
    return

# 统计LAI绝对差的误差，绘制误差统计直方图
def diffLAI_histogram():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    Tem_improvedArray = []
    Spa_improvedArray = []
    for i in range(1, 47):
        tem_data = np.loadtxt('./Daily_cache/0522/Tem_LAI/LAI_%s' % i)
        spa_data = np.loadtxt('./Daily_cache/0522/Spa_LAI/LAI_%s' % i)
        Tem_improvedArray.append(tem_data)
        Spa_improvedArray.append(spa_data)
    
    Tem_improvedLAI = np.array(Tem_improvedArray)
    Spa_improvedLAI = np.array(Spa_improvedArray)

    # 相对标准数据的绝对差
    Ina = (StandLAI - InaccurateLAI) / 10
    Tem = (StandLAI - Tem_improvedLAI) / 10
    Spa = (StandLAI - Spa_improvedLAI) / 10
    # i = 0
    print(Ina.shape, np.max(Ina), np.min(Ina))
    print(Ina.shape, np.max(Tem), np.min(Tem))
    print(Ina.shape, np.max(Spa), np.min(Spa))
  
    print(np.nonzero(Spa < -4))
    print(StandLAI[:, 337,133])
    print(Spa_improvedLAI[:, 337,133])
    print(Tem_improvedLAI[:, 337,133])
    print(InaccurateLAI[:, 238,440])
    
    

    # 绘制误差的分布密度直方图
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(Ina.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.8, label='Inaccurate')
    ax.hist(Tem.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Temporal')
    ax.hist(Spa.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Spatial')

    
    ax.set_xlabel('Absolute Difference', fontsize=15, family='Times New Roman')
    ax.set_ylabel('Density', fontsize=15, family='Times New Roman')
    ax.legend(prop={'size':15, 'family':'Times New Roman'})
    fig.tight_layout()
    plt.xticks( family='Times New Roman', fontsize=15)
    plt.yticks( family='Times New Roman', fontsize=15)
    plt.savefig('./Daily_cache/0530/diffLAI_histogram', dpi=300)
    plt.show()

# 修改模拟数据特别异常的点
def update():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    print(StandLAI[:, 238,440])
    # StandLAI[-8:, 238,440] = (4,4,4,4,4,4,4,4)
    # print(StandLAI[:, 238,440])
    InaccurateLAI[-8:, 238,440] = (4,4,4,4,4,4,4,4)
    # np.save('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard', StandLAI)
    # np.save('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70)', InaccurateLAI)

    # Err_weight = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight.npy')
    # # print(Err_value[-8:, 238,440])
    # # print(Err_weight[43,239,216])

    # Err_weight[-8:, 238,440] = (10,10,10,10,10,10,10,10)
    # np.save('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/Err_weight', Err_weight)
    return

