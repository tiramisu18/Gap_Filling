import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
from scipy.stats import gaussian_kde

def render_LAI (data, title='LAI', issave=False, savepath=''):
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

def render_Img (data, title='Title', issave=False, savepath='', color=plt.cm.jet):
    plt.imshow(data, cmap = color)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    # plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()


# color and marker  = False or ['xxx']; lineStyle = [] or ['xxx]
def draw_polt_Line (x, obj, savePath = '', issave = False, loc = 0):
    color_arr = ['#548bb7', '#958b8c', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c', '#016382', '#dd8146', '#a4ac80', '#d9b15c', '#1f8a6f', '#987b2d']
    marker_arr = ['s', 'o', '.', '^', ',', 'v', '8', '*', 'H', '+', 'x', '_']
    if obj['color'] : color_arr = obj['color']
    if obj['marker'] : marker_arr = obj['marker']
    if obj['size'] : plt.figure(figsize=(obj['size']['width'], obj['size']['height']))
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

def polt_Line_twoScale (x, obj, savePath = '', issave = False, loc = 0):
    color_arr = ['#548bb7', '#958b8c', '#bfdb39', '#ffe117', '#fd7400', '#7ba79c', '#016382', '#dd8146', '#a4ac80', '#d9b15c', '#1f8a6f', '#987b2d']
    marker_arr = ['o', '.', '^', 's', ',', 'v', '8', '*', 'H', '+', 'x', '_']
    if obj['color'] : color_arr = obj['color']
    if obj['marker'] : marker_arr = obj['marker']
    fig, ax1 = plt.subplots()
    ax1.set_title(obj['title'], family='Times New Roman', fontsize=18)   
    ax1.set_xlabel(obj['xlable'], fontsize=15, family='Times New Roman') 
    ax1.set_ylabel(obj['ylable'], fontsize=15, family='Times New Roman')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Weight', color=color, fontsize=15, family='Times New Roman')  # we already handled the x-label with ax1
    # ax2.plot(x, obj['line'][4], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    line_arr = []
    count_i = -1
    for idx in range(0, len(obj['line'][0])):
        count_i += 1 
        line_arr.append((ax1.plot(x,obj['line'][0][idx], label='count', color=color_arr[count_i],  marker=marker_arr[count_i], markersize=3))[0])
    for idx in range(0, len(obj['line'][1])):
        count_i += 1 
        line_arr.append((ax2.plot(x,obj['line'][1][idx], label='count', color=color_arr[count_i],  marker=marker_arr[count_i], markersize=3, linestyle='dashed'))[0])
    
    # for i in range(0, 2): 
    #     for j in range(0, len(obj['line'][i])):  
    #         line_arr.append()         
    #     if i < ls_len : 
    #         line_arr.append((ax1.plot(x,obj['line'][0][i], label='count', color=color_arr[i],  marker=marker_arr[i], markersize=3, linestyle=obj['lineStyle'][i]))[0])
    #     else: 
    #         line_arr.append((ax1.plot(x,obj['line'][0][i], label='count', color=color_arr[i],  marker=marker_arr[i], markersize=3))[0])
    plt.legend(
    (line_arr), 
    (obj['le_name']),
    loc = loc, prop={'size':15, 'family':'Times New Roman'},
    )
    if issave :plt.savefig(savePath, dpi=300)
    plt.show()

def draw_Line (x, y1, y2, y3, y4, savePath, issave, title = 'title'):
    # aa = np.arange(1, 21, 1)
    # plt.xticks(aa)
    # plt.figure(figsize=(15, 6)) #宽，高
    plt.title(title, family='Times New Roman', fontsize=18)   
    plt.xlabel('day', fontsize=15, family='Times New Roman') 
    plt.ylabel('LAI', fontsize=15, family='Times New Roman')
    line1=plt.plot(x,y1, label='count', color='gray',  marker='o', markersize=3, linestyle= 'dashed')
    line2=plt.plot(x,y2, label='count', color='#bfdb39',  marker='.', markersize=3)
    line3=plt.plot(x,y3, label='count', color='#ffe117',  marker='^', markersize=3)
    line4=plt.plot(x,y4, label='count', color='#fd7400',  marker='s', markersize=3)
    plt.legend(
    (line1[0],  line2[0],  line3[0],  line4[0]), 
    # ('Original', 'Filling1', 'Filling2', 'Fil_NOM'),
    ('Original', 'Tem', 'Spa', 'Fil'),
    loc = 2, prop={'size':15, 'family':'Times New Roman'},
    )
    if issave :plt.savefig(savePath, dpi=300)
    plt.show()

def draw_one_plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, '#fd7400')
    # plt.plot(x, y, 'r--', x, t**2, 'bs', t, t**3, 'g^')
    # plt.plot(t, y1, '#fd7400', t, y2, '#bfdb39')
    ax.set(xlabel='day', ylabel='LAI', title='title')
    # ax.grid()

    # fig.savefig("test.png")
    plt.show() 

# 散点图
def density_scatter_plot(x, y):
    # x = np.random.normal(size=500)
    # y = x * 3 + np.random.normal(size=500)
    # print(x, y)
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx] 
    fig, ax = plt.subplots()
    plt.scatter(x, y,c=z,  s=20,cmap='Spectral')
    plt.colorbar()
    plt.show()