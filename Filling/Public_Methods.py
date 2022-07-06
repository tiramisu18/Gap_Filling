import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
from scipy import stats


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

def render_Img (data, title='', issave=False, savepath='', color=plt.cm.jet, axisType = 'off'):
    plt.imshow(data, cmap = color)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    plt.axis(axisType)
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
    
    fig, ax1 = plt.subplots(figsize=(obj['size']['width'], obj['size']['height']))
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
# 散点图参数计算
def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    #a、b、r
    # print(slope, intercept,"r", r_value,"r-squared", r_value**2)
    return [slope, intercept, r_value**2]
    
# 散点图
def density_scatter_plot(x, y, url):
    calRMSE = np.sqrt((1/len(x))* np.sum(np.square(x - y)))
    parameter = rsquared(x, y)
    print('RMSE, a, b, R2', calRMSE, parameter)
    y2 = parameter[0] * x + parameter[1]

    plt.scatter(x, y, color='#bfdb39')
    plt.ylabel(f'{type} LAI', fontsize=15, family='Times New Roman')
    plt.xlabel('GBOV LAI', fontsize=15, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    # parameter = np.polyfit(x, y, deg=1)
    # print(parameter)
    
    plt.plot(x, y2, color='#ffe117', linewidth=1, alpha=1)
    plt.plot((0, 7), (0, 7),  ls='--',c='k', alpha=0.8, label="1:1 line")
    plt.savefig(url, dpi=300)
    plt.show()

# Discrete distribution as horizontal bar chart
def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    # category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))
    category_colors = ['#e44f35','#fcbb6b', '#fdffbe', '#b3df72', '#3faa5a']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    # ax.set_ylabel()
    # ax.set_xlabel('Day', fontsize=15, family='Times New Roman')
    # plt.xlabel('GBOV LAI', fontsize=15, family='Times New Roman')

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', fontsize=5, family='Times New Roman')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', prop={'size':12, 'family':'Times New Roman'})

    return fig, ax