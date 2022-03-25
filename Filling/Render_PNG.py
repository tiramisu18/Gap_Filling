import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor

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
    plt.imshow(data, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

# LAI_Simu_Err = np.load('../Simulation/Simulation_Dataset/Method_2/LAI_Simu_addErr(0-70).npy')
# for i in range(0, 46):
#     render_LAI(LAI_Simu_Err[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s' % (i+1))
#     # render_LAI(LAI_Simu_Err[i], '2018_%d_Err'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s_addErr' % (i+1))


# Err_value = np.load('../Simulation/Simulation_Dataset/Method_2/Err_value.npy')
# Err_peren = np.load('../Simulation/Simulation_Dataset/Method_2/Err_peren.npy')
# for i in range(3,4):
#     render_Img(Err_value[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/2018_%s_errValue' % (i+1))
#     # render_Img(Err_peren[i], '2018_%d'% (1 + i*8),True, './Daily_cache/0316/2018_%s_errperen' % (i+1))
#     # render_LAI(LAI_Simu_Err[i], '2018_%d_Err'% (1 + i*8),True, './Daily_cache/0316/Step2_LAI_AddErr/2018_%s_addErr' % (i+1))

# 填补后的数据
for i in range(2,3):
    data_array = np.loadtxt('../Simulation/Filling/2018_%s' % i)
    render_LAI(data_array, '2018_%d_Filling'% (1 + (i-1)*8),True, './Daily_cache/0316/Step2_LAI_Filling/2018_%s' % (i))


