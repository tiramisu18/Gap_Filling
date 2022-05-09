import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import random
import scipy.io as scio
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

# 随机增加误差后生成的误差数据集、误差百分比数据集、误差值数据集
def add_ErrDataSet():
    # LAI_Err_Peren = [[[0.0] * 500] * 500] * 46 # 生成浮点型的矩阵
    # np.save('./Simulation_Dataset/Err_zero(double)', LAI_Err_Peren)
    LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_noErr(0-7).npy')
    Err_Peren = np.load('./Simulation_Dataset/Err_zero(int).npy')
    Err_value = np.load('./Simulation_Dataset/Err_zero(double).npy')
    for day in range(0,46):
        print(day)
        x = int_random(0, 499, 50000)
        y = int_random(0, 499, 50000)    
        for i in range(0, 50000): # 选取20%的像元            
            # print(day, x[i], y[i])
            L_ori = LAI_Simu[day][x[i]][y[i]]
            if Err_Peren[day][x[i]][y[i]] == 0 and L_ori <= 7 and L_ori > 0:  
                err = round(random.uniform(-1.5, 1.5),1)          
                LAI_addErr = LAI_Simu[day][x[i]][y[i]] + err
                if LAI_addErr > 7:
                    LAI_Simu[day][x[i]][y[i]] = 7
                    err = 7 - L_ori
                elif LAI_addErr < 0:
                    LAI_Simu[day][x[i]][y[i]] = 0
                    err = -L_ori
                else:
                    LAI_Simu[day][x[i]][y[i]] = round(LAI_addErr,2)
                try :
                    Err_Peren[day][x[i]][y[i]] = int(round(abs(err) / L_ori, 2) * 100)
                    Err_value[day][x[i]][y[i]] = err
                except:
                    print(err, L_ori)
    
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/Err_peren', Err_Peren)       
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/Err_value', Err_value) 

    for idx in range(0, 46):
        print(idx)
        for i in range(0, 500):
            for j in range(0,500):
                if LAI_Simu[idx][i][j] <= 7 : 
                    LAI_Simu[idx][i][j] = LAI_Simu[idx][i][j] * 10
    np.save('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_addErr(0-70)', LAI_Simu)

def add_ErrDataSet_matrix():
    LAI_Simu = np.array(np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_noErr(0-7).npy'))
    relativeErr_value = np.random.uniform(-80,80,(46,500,500)) # 相对误差比例   
    # print(relativeErr_value)
    Err_value = np.around(LAI_Simu * relativeErr_value * 0.01, 2)
    # np.set_printoptions(precision = 1)
    # print(Err_value)
    LAI_ma = ma.masked_values(ma.masked_greater(LAI_Simu, 7),0)
    LAI_addErr = np.around(LAI_ma + Err_value,2)
    # 处理小于0的部分
    pos1 = LAI_addErr.__lt__(0)
    Err_value[pos1] = LAI_Simu[pos1] * -1
    LAI_addErr[pos1] = 0
    # 处理大于7的部分
    zero = np.zeros(46*500*500, dtype=int).reshape(46, 500, 500)
    pos2 = np.logical_and(LAI_addErr > 7, LAI_addErr < 250)
    zero[pos2] = 7
    dif = np.around(zero - LAI_Simu,2)
    Err_value[pos2] = dif[pos2]
    LAI_addErr[pos2] = 7
    # 检验是否有不符合的数值
    # aa = np.nonzero(LAI_addErr > 7)
    # bb = np.nonzero(LAI_addErr < 0)
    # print(aa,bb)
    # 放大数值至0-70
    LAI_addErr = LAI_addErr * 10   
    # 求误差百分比
    Err_Peren = np.around(np.abs(Err_value) / LAI_ma * 100, 0)
    # 将原始掩膜的位置再补回、原始掩膜位置的误差值设为0
    pos3 = np.logical_or(LAI_Simu > 7, LAI_Simu == 0)
    Err_value[pos3] = 0
    Err_Peren[pos3] = 0
    LAI_addErr[pos3] = LAI_Simu[pos3]
    # 找出所有为-0的位置，设置为0
    pos5 = LAI_addErr.__eq__(-0)
    LAI_addErr[pos5] = 0
    # 将掩膜数组转为非掩膜类型，否则存储会报错
    LAI_addErr = np.array(LAI_addErr)
    Err_Peren = np.array(Err_Peren)
    np.save('./Simulation_Dataset/LAI/Simu_Method_3/Err_peren', Err_Peren)       
    np.save('./Simulation_Dataset/LAI/Simu_Method_3/Err_value', Err_value)
    np.save('./Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70)', LAI_addErr)

# LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_noErr(0-7).npy')
LAI_Simu = np.load('./Simulation_Dataset/LAI/Simu_Method_2/LAI_Simu_Step2.npy')
LAI_addErr = np.load('./Simulation_Dataset/LAI/Simu_Method_3/LAI_Simu_addErr(0-70).npy')

# err_value = np.load('./Simulation_Dataset/LAI/Simu_Method_2/Err_value.npy')


x_v = 50
y_v = 50
# (0,3) (2,1) （2，2） (499, 499)
aa = LAI_Simu[:, x_v, y_v] / 10
bb = LAI_addErr[:, x_v, y_v] / 10

# print(aa,bb)

draw_polt_Line(np.arange(0, 361, 8),{
    'title': 'LAI_addErr',
    'xlable': 'Day',
    'ylable': 'LAI',
    'line': [aa, bb, bb],
    'le_name': ['Simu', 'addErr', '','Fil'],
    'color': ['#bfdb39', 'gray','#fd7400', '#1f8a6f', '#548bb7','gray', '#bfdb39',],
    'marker': [',', ',','^',],
    'lineStyle': ['dashed', 'dashed','']
    },'../Filling/Daily_cache/0506/lai_addErr', True, 2)




