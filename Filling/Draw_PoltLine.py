import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor

# color and marker  = False or ['xxx']; lineStyle = [] or ['xxx]
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
        plt.plot(x, obj['line'][0], '#548bb7')
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