import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy.ma as ma

def render_LAI (data, title='', issave=False, savepath=''):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title(title, family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = 'Times New Roman'
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','1','2','3','4','5','6','7','250'])   
    # cbar.ax.tick_params(labelsize=13, bottom=True)
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


# 计算数据提升之后整个tile46期与原始含有误差数据的RMSE以及RMSE的密度分布直方图
def calRMSE_allTile():
    spa_array, tem_array, imp_array = [], [], []
    for i in range(1, 47):
        # spa_array.append(np.load(f'../Improved/Improved_SimuData/Spatial/LAI_{i}.npy'))
        # tem_array.append(np.load(f'../Improved/Improved_SimuData/Temporal/LAI_{i}.npy'))
        imp_array.append(np.load(f'../Improved/Improved_SimuData/Improved/LAI_{i}.npy'))
    # spatialLAI = np.array(spa_array)
    # temporalLAI = np.array(tem_array)
    improvedLAI = np.array(imp_array)
    standLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    saveURL = './Daily_cache/Final_Image/Simulated_LAI/RMSE'
    # Spatial Temporal
    # calRMSE_spa = np.sqrt((1/len(spatialLAI))* np.sum(np.square(standLAI - spatialLAI), axis=0)) / 10
    # render_Img(calRMSE_spa, title='RMSE', savepath=f'{saveURL}/RMSE_Spatial', issave=True)
    # calRMSE_tem = np.sqrt((1/len(temporalLAI))* np.sum(np.square(standLAI - temporalLAI), axis=0)) / 10
    # render_Img(calRMSE_tem, title='RMSE', savepath=f'{saveURL}/RMSE_Temoral', issave=True)  
    # print(np.mean(calRMSE_tem), np.mean(calRMSE_spa))

    # Improved Inaccurate
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    calRMSE_inacc = np.sqrt((1/len(improvedLAI))* np.sum(np.square(standLAI - InaccurateLAI), axis=0)) / 10
    calRMSE_imp = np.sqrt((1/len(improvedLAI))* np.sum(np.square(standLAI - improvedLAI), axis=0)) / 10
    # print(np.mean(calRMSE_inacc), np.mean(calRMSE_imp))
    # render_Img(calRMSE_imp, title='RMSE', savepath=f'{saveURL}/RMSE_Improved', issave=True)  
    # render_Img(calRMSE_inacc, title='RMSE', savepath=f'{saveURL}/RMSE_Inaccurate', issave=True)
    data = [calRMSE_inacc, calRMSE_imp]
    count = 2
    fig, axs = plt.subplots(1, count)
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = 'Times New Roman'
    # fig.suptitle('Multiple images')
    images = []

    for j in range(count):
            # Generate data with a range that varies from one plot to the next.
            # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
        images.append(axs[j].imshow(data[j], cmap = plt.cm.coolwarm))
        axs[j].axis('off')
            # axs[i, j].width = 

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    # vmax = max(image.get_array().max() for image in images)
    vmax = 1
    norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs,  fraction=.1)
    plt.savefig(f'{saveURL}/RMSE_combine', dpi=300)
    plt.show()

    # histogram
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(calRMSE_inacc.flatten(), density=True, histtype="stepfilled", bins=1000, color='#d5e6ae', label='Inaccurate')
    ax.hist(calRMSE_imp.flatten(), density=True, histtype="stepfilled", bins=1000, alpha=0.8, color='#b8b1d1', label='Improved')

    
    ax.set_xlabel('RMSE', fontsize=20, family='Times New Roman')
    ax.set_ylabel('Density', fontsize=20, family='Times New Roman')
    ax.legend(framealpha=0, prop={'size':20, 'family':'Times New Roman'}, loc=1, bbox_to_anchor=(0.9, 0.75), labelspacing=1)
    fig.tight_layout()
    ax.set(xlim=(0.05, 1.1), ylim=(0, 6), yticks=np.arange(0, 7, 2))
    plt.xticks( family='Times New Roman', fontsize=20)
    plt.yticks( family='Times New Roman', fontsize=20)
    plt.savefig(f'{saveURL}/RMSE_histogram', dpi=300)
    plt.show()

# 计算单独一期整个tile的LAI差
def diff_LAI():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')

    i = 23
    tem_data = np.load(f'../Improved/Improved_SimuData/Temporal/LAI_{i+1}.npy')
    spa_data = np.load(f'../Improved/Improved_SimuData/Spatial/LAI_{i+1}.npy')
    imp_data = np.load(f'../Improved/Improved_SimuData/Improved/LAI_{i+1}.npy')


    render_LAI(StandLAI[i], issave=True, savepath='./Daily_cache/Final_Image/Simulated_LAI/Standard_LAI')
    render_LAI(InaccurateLAI[i], issave=True, savepath='./Daily_cache/Final_Image/Simulated_LAI/Inaccurate_LAI')
    render_LAI(tem_data, issave=True, savepath='./Daily_cache/Final_Image/Simulated_LAI/Tem_LAI')
    render_LAI(spa_data, issave=True, savepath='./Daily_cache/Final_Image/Simulated_LAI/Spa_LAI')
    render_LAI(imp_data, issave=True, savepath='./Daily_cache/Final_Image/Simulated_LAI/Improved_LAI')

    # 相对标准数据的绝对差
    Ina = (StandLAI[i]-InaccurateLAI[i])/10
    # Tem = (StandLAI[i]-tem_data)/10
    # Spa = (StandLAI[i]-spa_data)/10
    Imp = (StandLAI[i]-imp_data)/10
    data = [Ina, Imp]
    count = 2
    fig, axs = plt.subplots(1, count)
    # fig.suptitle('Multiple images')
    images = []

    for j in range(count):
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
    plt.savefig('./Daily_cache/Final_Image/Simulated_LAI/diff', dpi=300)
    plt.show()
    # print(np.mean(np.abs(Ina)), np.mean(np.abs(Tem)), np.mean(np.abs(Spa)), np.mean(np.abs(Imp)))
    print(np.mean(np.abs(Ina)), np.mean(np.abs(Imp)))
    
    return

# 统计LAI绝对差的误差, 绘制误差统计直方图
def diffLAI_histogram():
    StandLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Standard.npy')
    InaccurateLAI = np.load('../Simulation/Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')
    Tem_improvedArray = []
    Spa_improvedArray = []
    for i in range(1, 47):
        tem_data = np.loadtxt('../Improved/Improved_SimuData/Tem_LAI/LAI_%s' % i)
        # spa_data = np.loadtxt('../Improved/Improved_SimuData/Spa_LAI/LAI_%s' % i)
        Tem_improvedArray.append(tem_data)
        # Spa_improvedArray.append(spa_data)
    
    Tem_improvedLAI = np.array(Tem_improvedArray)
    # Spa_improvedLAI = np.array(Spa_improvedArray)

    # 相对标准数据的绝对差
    Ina = (StandLAI - InaccurateLAI) / 10
    Tem = (StandLAI - Tem_improvedLAI) / 10
    # Spa = (StandLAI - Spa_improvedLAI) / 10
    # i = 0
    # print(Ina.shape, np.max(Ina), np.min(Ina))
    # print(Ina.shape, np.max(Tem), np.min(Tem))
    # print(Ina.shape, np.max(Spa), np.min(Spa))
      
    
    # 绘制误差的分布密度直方图
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(Ina.flatten(), density=True, histtype="stepfilled", bins=80, alpha=0.8, label='Inaccurate')
    # ax.hist(Tem.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Temporal')
    # ax.hist(Spa.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Spatial')
    ax.hist(Tem.flatten(), density=True, histtype="stepfilled", bins=80, alpha=0.6, label='Improved')

    
    ax.set_xlabel('Absolute Difference', fontsize=15, family='Times New Roman')
    ax.set_ylabel('Density', fontsize=15, family='Times New Roman')
    ax.legend(prop={'size':15, 'family':'Times New Roman'})
    fig.tight_layout()
    plt.xticks( family='Times New Roman', fontsize=15)
    plt.yticks( family='Times New Roman', fontsize=15)
    plt.savefig('./Daily_cache/0718/diffLAI_histogram', dpi=300)
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

