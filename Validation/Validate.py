import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import random
import scipy.io as scio

# 随机增加误差后生成的误差数据集以及误差百分比数据集
def get_ErrDataSet():
    # # LAI_Err_Peren = [[[0] * 500] * 500] * 46
    # # np.save('./Simulation_Dataset/Err_zero', LAI_Err_Peren)
    LAI_Simu = np.load('./Simulation_Dataset/LAI_Simu.npy')
    LAI_Err_Peren = np.load('./Simulation_Dataset/Err_zero.npy')
    for day in range(0,46):
        print(day)
        x = int_random(0, 499, 50000)
        y = int_random(0, 499, 50000)    
        for i in range(0, 50000):
            err = round(random.uniform(0.1, 1.5),1)
            # print(day, x[i], y[i])
            L_ori = LAI_Simu[day][x[i]][y[i]]
            if LAI_Err_Peren[day][x[i]][y[i]] == 0 and L_ori <= 7 and L_ori > 0:            
                LAI_addErr = LAI_Simu[day][x[i]][y[i]] + err
                if LAI_addErr > 7:
                    LAI_Simu[day][x[i]][y[i]] = 7
                    err = 7 - L_ori
                else:
                    LAI_Simu[day][x[i]][y[i]] = round(LAI_addErr,2)
                try :
                    LAI_Err_Peren[day][x[i]][y[i]] = int(round(err / L_ori, 2) * 100)
                except:
                    print(err, L_ori)
    np.save('./Simulation_Dataset/LAI_addErr', LAI_Simu)
    np.save('./Simulation_Dataset/Err_peren', LAI_Err_Peren)   
    # 测试上面程序是否正确运行的测试数据
    # LAI_Simu = [[[2.1,4],[4,8]], [[6.2,4],[4,12]]]
    # LAI_Err_Peren = [[[0,0], [0,0]],[[0,0], [0,0]]]
    # for day in range(0,2):
    #     print(day)
    #     x = [0,1]
    #     y = [0,0]    
    #     for i in range(0, 2):
    #         err = 4.8
    #         if LAI_Err_Peren[day][x[i]][y[i]] == 0:
    #             L_ori = LAI_Simu[day][x[i]][y[i]]
    #             LAI_addErr = LAI_Simu[day][x[i]][y[i]] + err
    #             if LAI_addErr > 7:
    #                 LAI_Simu[day][x[i]][y[i]] = 7
    #                 err = 7 - L_ori
    #             else:
    #                 LAI_Simu[day][x[i]][y[i]] = round(LAI_addErr,2)
    #             LAI_Err_Peren[day][x[i]][y[i]] = int(round(err / L_ori, 2) * 100)
    # print(LAI_Simu)
    # print(LAI_Err_Peren)

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

def render_LAI_Simu (data, title='Image', issave=False, savepath=''):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,1,2,3,4,5,6,7,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title(title, family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','1','2','3','4','5','6','7','250'])
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

def render_Img (data, title='Img', issave=False, savepath=''):
    plt.imshow(data, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

# 生成一个包含n个介于a和b之间随机整数的数组（可重复）
def int_random(a, b, n) :
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        a_list.append(d_int)
        # if(d_int not in a_list) :
        #     a_list.append(d_int)
        # else :
        #     pass
    return a_list

def draw_polt_Line (x, obj, savePath = '', issave = False, loc = 0):
    color_arr = ['#548bb7', '#958b8c', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c', '#016382', '#dd8146', '#a4ac80', '#d9b15c', '#1f8a6f', '#987b2d']
    marker_arr = ['o', '.', '^', 's', ',', 'v', '8', '*', 'H', '+', 'x', '_']
    if obj['color'] : color_arr = obj['color']
    if obj['marker'] : marker_arr = obj['marker']
    plt.title(obj['title'], family='Times New Roman', fontsize=18)   
    plt.xlabel(obj['xlable'], fontsize=15, family='Times New Roman') 
    plt.ylabel(obj['ylable'], fontsize=15, family='Times New Roman')
    obe_len = len(obj['line'])
    if obe_len == 1:
        plt.plot(x, obj['line'][0], '#fd7400')
        if issave :plt.savefig(savePath, dpi=300)
        plt.show() 
    else:
        line_arr = []
        ls_len = len(obj['lineStyle'])
        for i in range(0, obe_len):            
            if i < ls_len : 
                line_arr.append((plt.plot(x,obj['line'][i], label='count', color=color_arr[i],  marker=marker_arr[i], markersize=3, linestyle=obj['lineStyle'][i]))[0])
            else: 
                line_arr.append((plt.plot(x,obj['line'][i], label='count', color=color_arr[i],  marker=marker_arr[i], markersize=3))[0])
        plt.legend(
        (line_arr), 
        (obj['le_name']),
        loc = loc, prop={'size':15, 'family':'Times New Roman'},
        )
        if issave :plt.savefig(savePath, dpi=300)
        plt.show()

# Vage_All = np.load('./Simulation_Dataset/Vege_data.npz', allow_pickle=True)
# LandCover = np.load('./Simulation_Dataset/LandCover.npy')
# LAI_Ori = np.load('./Simulation_Dataset/LAI_Ori.npy')
LAI_Simu = np.load('./Simulation_Dataset/LAI_Simu.npy')
LAI_addErr = np.load('./Simulation_Dataset/LAI_addErr.npy')
Err = np.load('./Simulation_Dataset/Err_peren.npy')

# dataNew = 'LAI_Err.mat'
# scio.savemat(dataNew, {'Err':Err})

# render_Img(Err[5])
print(LAI_Simu[4][2][1])
print('----')
print(LAI_addErr[4][2][1])
print('----')
print(Err[0])

aa = []
bb = []
# (0,3) (2,1) （2，2） (499, 499)
for i in range(0, 46):
    # render_Img(Err[i])
#     render_LAI_Simu(LAI_Simu[i], 'Simu_%s'%i)
#     render_LAI_Simu(LAI_addErr[i], 'AddErr_%s'%i)   
    aa.append(LAI_Simu[i][2][1])
    bb.append(LAI_addErr[i][2][1])

draw_polt_Line(np.arange(1, 47, 1),{
    'title': 'BX',
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': [aa, bb],
    'le_name': ['Sium', 'addErr', 'Spa', 'Fil'],
    'color': ['#ffe117', '#fd7400', '#1f8a6f', '#548bb7','gray', '#bfdb39',],
    'marker': False,
    'lineStyle': []
    },'./Daily_cache/0309/vegeType_mirror',False, 2)


# get_ErrDataSet()




