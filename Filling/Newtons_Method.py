import numpy as np
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import time
from scipy import optimize

def draw_function_dimensionTwo(): 
    x = np.arange(-100, 100, 0.001)
    y = x ** 3 / (3 * x + 1)
    
    plt.figure()
    plt.plot(x, y, linestyle='--', color='red')
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.show()


def draw_function_dimensionThree(z): 

    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    # x=np.arange(-0.1,1.2,0.01)
    # y=np.arange(10,50,5)
    x=np.arange(0.0001,5,0.015)
    y=np.arange(0,5,0.03)
    x, y = np.meshgrid(x, y) # 网格的创建，生成二维数组，这个是关键
    # z=x**2+y**2
    # z = (x-1)**2+(y-5)**2
    z = 8.0*(-1 + 0.0833333333333333*(19.0*x + 14.0*y)/(x + y))**2 + 12.5*(-1 + 0.0666666666666667*(19.0*x + 15.0*y)/(x + y))**2 + 32.0*(-1 + 0.0416666666666667*(19.0*x + 28.0*y)/(x + y))**2 + 18.0*(-1 + 0.0555555555555556*(20.0*x + 17.0*y)/(x + y))**2 + 16.0555555555556*(-1 + 0.0588235294117647*(20.0*x + 18.0*y)/(x + y))**2 + 29.3888888888889*(-1 + 0.0434782608695652*(20.0*x + 24.0*y)/(x + y))**2 + 9.38888888888889*(-1 + 0.0769230769230769*(21.0*x + 15.0*y)/(x + y))**2 + 20.0555555555556*(-1 + 0.0526315789473684*(21.0*x + 19.0*y)/(x + y))**2 + 22.2222222222222*(-1 + 0.05*(21.0*x + 21.0*y)/(x + y))**2 + 29.3888888888889*(-1 + 0.0434782608695652*(21.0*x + 23.0*y)/(x + y))**2 + 37.5555555555556*(-1 + 0.0384615384615385*(22.0*x + 26.0*y)/(x + y))**2 + 26.8888888888889*(-1 + 0.0454545454545455*(23.0*x + 23.0*y)/(x + y))**2 + 46.7222222222222*(-1 + 0.0344827586206897*(23.0*x + 28.0*y)/(x + y))**2 + 50.0*(-1 + 0.0333333333333333*(23.0*x + 30.0*y)/(x + y))**2 + 50.0*(-1 + 0.0333333333333333*(24.0*x + 27.0*y)/(x + y))**2 + 40.5*(-1 + 0.037037037037037*(24.0*x + 27.0*y)/(x + y))**2 + 46.7222222222222*(-1 + 0.0344827586206897*(25.0*x + 32.0*y)/(x + y))**2 + 34.7222222222222*(-1 + 0.04*(26.0*x + 25.0*y)/(x + y))**2
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.plot_surface(x, y, z, cmap='rainbow')
    # plt.savefig('./Daily_cache/0413/Weight_graph_3', dpi=300)
    plt.show()


def Newtons_Simulated_Dataset(Tem, Spa, Raw):
    Tem = [15.0, 18.0, 24.0, 19.0, 28.0, 14.0, 21.0, 26.0, 23.0, 15.0, 17.0, 23.0, 27.0, 32.0, 27.0, 28.0, 25.0, 30.0]
    Spa = [19.0, 20.0, 20.0, 21.0, 19.0, 19.0, 21.0, 22.0, 21.0, 21.0, 20.0, 23.0, 24.0, 25.0, 24.0, 23.0, 26.0, 23.0]
    Raw = [15.0, 17.0, 23.0, 19.0, 24.0, 12.0, 20.0, 26.0, 23.0, 13.0, 18.0, 22.0, 30.0, 29.0, 27.0, 29.0, 25.0, 30.0]
    x, y= symbols('x y')
    dataSum = 0
    for i in range(0, len(Tem)):
        final = (Spa[i] * x + Tem[i] * y) / (x + y)
        dataSum = dataSum + ((final - Raw[i])**2)
    Fxy = dataSum / len(Tem)
    # print(Fxy)
    draw_function_dimensionThree(Fxy)
    print(Fxy.evalf(subs = {'x': 0.0490005, 'y': 1}))
    print(Fxy.evalf(subs = {'x': 1.15, 'y': 12.5141}))
    print(Fxy.evalf(subs = {'x': 0.074, 'y': 0.926}))
    # return
    f1 = diff(Fxy,x)
    f2 = diff(Fxy,y)

    F = Matrix([f1,f2])
    def hessian(*para):
        all = []
        for f_i in para[0]:
            one = []
            for o_i in para[1]:
                one.append(diff(f_i, o_i))
            all.append(one)
            # sp.diff(f,o).evalf(subs ={'x2':6})
        # return np.mat(all).reshape(3,3)
        return Matrix(all)

    J = hessian((f1, f2), (x, y))

    n=1
    x0=np.mat([0.1,0.1]).reshape(2,1)
    # mar = J.subs(dict(x=x0[0,0], y=x0[1,0], z=x0[2,0]))
    # print(mar, mar.inv())

    E=1
    while n < 10 and E > 1e-4:
        dic = dict(x=x0[0,0], y=x0[1,0])
        x1=x0-J.subs(dic).inv()*F.subs(dic)
        step1 = (diag(x1[0,0], x1[1, 0])).inv() * abs(x1-x0)
        # print('step1' , step1)
        # E=max(abs(x1-x0)/x1)
        E=max(step1)
        x0=x1
        n=n+1

    # print(x1, n)

    calValue = Fxy.evalf(subs = {'x': x1[0,0], 'y': x1[1,0]})
    print({'X_Valus': x1, 'Iteration_Count': n, 'Multinomial_Value': round(calValue, 6)})


# 牛顿迭代法求极值
def Newtons_Iteration_xyz():
    # syms x y z
    x, y, z = symbols('x y z')
    Fxyz = (x-1)**2+(y-5)**2+(z-100)**2
    # Fxyz = (((25 * x + 37 * y + 30 * z) / (x + y + z) - 30)** 2 + ((30 * x + 27 * y + 20 * z) / (x + y + z) - 20)** 2) / 2
    # Fxyz = 4*(x + 1)**2 + 2*(y - 2)**2 + x + y + 10
    # Fxyz = x**2+2*y**2+3*z**2+2*x+4*y-6*z 

    f1 = diff(Fxyz,x)
    f2 = diff(Fxyz,y)
    f3 = diff(Fxyz,z)

    # F = np.mat([f1,f2,f3]).reshape(3,1) # 列表转矩阵
    F = Matrix([f1,f2,f3])
    # 海森矩阵
    def hessian(*para):
        all = []
        for f_i in para[0]:
            one = []
            for o_i in para[1]:
                one.append(diff(f_i, o_i))
            all.append(one)
            # sp.diff(f,o).evalf(subs ={'x2':6})  #对求导后的式子附值计算
        # return np.mat(all).reshape(3,3)
        return Matrix(all)

    # J_be=[[sp.diff(f1,x), sp.diff(f1, y), sp.diff(f1,z)],
    #    [sp.diff(f2,x), sp.diff(f2, y), sp.diff(f2,z)],
    #    [sp.diff(f3,x), sp.diff(f3, y), sp.diff(f3,z)]]

    # print(J_be)
    # J = np.mat(J_be).reshape(3,3)
    J = hessian((f1, f2, f3), (x, y, z))

    n=1
    x0 = np.mat([0,0,0]).reshape(3,1)
    E=1
    while n < 100 and E > 1e-4:
        dic = dict(x = x0[0,0], y = x0[1,0], z = x0[2,0])
        x1 = x0 - J.subs(dic).inv() * F.subs(dic) #牛顿迭代公式
        step1 = (diag(x1[0,0], x1[1, 0], x1[2, 0])).inv() * abs(x1-x0)
        # print('step1' , step1)
        # E=max(abs(x1-x0)/x1)
        E = max(step1)
        x0 = x1
        n = n + 1

    # sympy.subs()方法，将数学表达式中的变量或表达式的所有实例替换为其他变量或表达式或值。

    print(x1, n)
    calValue = Fxyz.evalf(subs = {'x': x1[0,0], 'y': x1[1,0], 'z':x1[2,0]})
    # Fxyz.evalf(subs = {'x': 1, 'y': 1, 'z':1})
    return {'X_Valus': x1, 'Iteration_Count': n, 'Multinomial_Value': round(calValue, 2)}


# 二元函数
def Newtons_Iteration_xy():
    x, y= symbols('x y')
    Fxy = (x-5)**2+(y-2)**2
    f1 = diff(Fxy,x)
    f2 = diff(Fxy,y)

    F = Matrix([f1,f2])
    def hessian(*para):
        all = []
        for f_i in para[0]:
            one = []
            for o_i in para[1]:
                one.append(diff(f_i, o_i))
            all.append(one)
            # sp.diff(f,o).evalf(subs ={'x2':6})
        # return np.mat(all).reshape(3,3)
        return Matrix(all)

    J = hessian((f1, f2), (x, y))

    n=1
    x0=np.mat([1,1]).reshape(2,1)
    # mar = J.subs(dict(x=x0[0,0], y=x0[1,0], z=x0[2,0]))
    # print(mar, mar.inv())

    E=1
    while n < 50 and E > 1e-4:
        dic = dict(x=x0[0,0], y=x0[1,0])
        x1=x0-J.subs(dic).inv()*F.subs(dic)
        step1 = (diag(x1[0,0], x1[1, 0])).inv() * abs(x1-x0)
        # print('step1' , step1)
        # E=max(abs(x1-x0)/x1)
        E=max(step1)
        x0=x1
        n=n+1

    # print(x1, n)

    calValue = Fxy.evalf(subs = {'x': x1[0,0], 'y': x1[1,0]})
    print({'X_Valus': x1, 'Iteration_Count': n, 'Multinomial_Value': round(calValue, 2)})


def Lagrange_simulated(Fxy, g, x, y, k):
    Tem = [15.0, 18.0, 24.0, 19.0, 28.0, 14.0, 21.0, 26.0, 23.0, 15.0, 17.0, 23.0, 27.0, 32.0, 27.0, 28.0, 25.0, 30.0]
    Spa = [19.0, 20.0, 20.0, 21.0, 19.0, 19.0, 21.0, 22.0, 21.0, 21.0, 20.0, 23.0, 24.0, 25.0, 24.0, 23.0, 26.0, 23.0]
    Raw = [15.0, 17.0, 23.0, 19.0, 24.0, 12.0, 20.0, 26.0, 23.0, 13.0, 18.0, 22.0, 30.0, 29.0, 27.0, 29.0, 25.0, 30.0]
    x, y, k= symbols('x y k')
    dataSum = 0
    for i in range(0, len(Tem)):
        final = (Spa[i] * x + Tem[i] * y) / (x + y)
        dataSum = dataSum + ((final - Raw[i])**2)
    Fxy = dataSum / len(Tem)
    g = x + y - 1
    #构造拉格朗日函数
    L=Fxy+k*g
    
    #求导
    dx = diff(L, x)   # 对x求偏导
    # print("dx=",dx)
    dy = diff(L,y)   #对y求偏导
    # print("dy=",dy)
    dk = diff(L,k)   #对k求偏导
    # print("dk=",dk)
    #求出个变量解
    m= solve([dx,dy,dk],[x,y,k])   
    print(m)
    #变量赋值
    print(Fxy.evalf(subs = {'x': 0.074, 'y': 0.926}))
    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in range(0, 10):  
        m= solve([dx,dy,dk],[x,y,k])   
        # m= nonlinsolve([dx,dy,dk],[x,y,k])   
        print(i)
    print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    

def Lagrange_simulated_martix(): # Fxy, g, x, y, k
    Tem1 = [15.0, 18.0, 24.0, 19.0, 28.0, 14.0, 21.0, 26.0, 23.0, 15.0]
    Spa = [19.0, 20.0, 20.0, 21.0, 19.0, 19.0, 21.0, 22.0, 21.0, 21.0]
    Raw = [15.0, 17.0, 23.0, 19.0, 24.0, 12.0, 20.0, 26.0, 23.0, 13.0]
    x, y, z, k= symbols('x y z k', real=True)
    # x, y, z, k= symbols('x y z k')
    dataSum1 = 0
    for i in range(0, len(Tem1)):
        final = (Spa[i] * x + Tem1[i] * y) / (x + y)
        dataSum1 = dataSum1 + ((final - Raw[i])**2)
        # dataSum1 = dataSum1 + abs(final - Raw[i])
    Fxy1 = dataSum1 / len(Tem1)

    Tem2 = [15.0, 18.0, 24.0, 19.0, 25.0, 14.0, 21.0, 26.0, 23.0, 15.0]
    dataSum2 = 0
    for i in range(0, len(Tem2)):
        final = (Spa[i] * x + Tem2[i] * y) / (x + y)
        dataSum2 = dataSum2 + ((final - Raw[i])**2)
        # dataSum2 = dataSum2 + abs(final - Raw[i])
    Fxy2 = dataSum2 / len(Tem2)
    
    g = x + y - 1
    # Lall = Fxy1+k*g
    Fxy = np.array([Fxy1, Fxy2])
    # Fxy = Matrix([Fxy1, x**2 + 8*y])
    # g = Matrix([g1, x+7*y])
   
    # Fxy = np.array([8*x*y*z, x**2 + 8*y])
    # g = x**2/a**2+y**2/b**2+z**2/c**2-1
  
    #构造拉格朗日函数
    L= Matrix(Fxy+k*g)
    # print(L)
    #求导
    dx = L.diff(x)   # 对x求偏导
    # print("dx=",dx.shape)
    dy = L.diff(y)   #对y求偏导
    # print("dy=",dy)
    # dz = L.diff(z)   #对y求偏导
    dk = L.diff(k)   #对k求偏导
    # print("dk=",dk)
    #求出个变量解
    
    # aa = np.array([dx,dy,dk]).reshape(3,2).T
    # print(aa)
    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    def myfun(dx,dy,dk):
        # print(1)
        res = solve([dx,dy,dk],(x,y,k),dict=True)
        return res
    stp1 = np.vectorize(myfun)
    # stp1 = np.frompyfunc(myfun,3,1)
    m = stp1(dx,dy,dk)
    print(m)
    print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # m= solve([dx,dy,dk],(x,y,k),dict=True)
    # print(m)
    # m= solve((dx,dy,dz,dk),(x,y,z,k),dict=True)
    # # print(res)
    # # bb = Matrix(aa)
    # # m= bb.solve([x,y,z,k])   
    # print(m)
# Lagrange_simulated_martix()
# Tem = [15.0, 18.0, 24.0, 19.0, 25.0, 14.0, 21.0, 26.0, 23.0, 15.0, 17.0, 26.0, 27.0, 25.0, 27.0, 28.0, 25.0, 30.0]
# Spa = [19.0, 20.0, 20.0, 21.0, 19.0, 19.0, 21.0, 22.0, 21.0, 21.0, 20.0, 23.0, 24.0, 25.0, 24.0, 23.0, 26.0, 23.0]
# Raw = [15.0, 17.0, 23.0, 19.0, 24.0, 12.0, 20.0, 26.0, 23.0, 13.0, 18.0, 22.0, 30.0, 29.0, 27.0, 29.0, 25.0, 30.0]
# x, y, k= symbols('x y k')
# dataSum = 0
# for i in range(0, len(Tem)):
#     final = (Spa[i] * x + Tem[i] * y) / (x + y)
#     dataSum = dataSum + ((final - Raw[i])**2)
# Fxy = dataSum / len(Tem)
# g = x + y - 1

# 拉格朗日求条件极值
def Lagrange(f, g, x, y, k):
    # x,y,z,k = symbols('x,y,z,k')
    # a,b,c=symbols('a,b,c')
    # f = 8*x*y*z
    # g = x**2/a**2+y**2/b**2+z**2/c**2-1
    #构造拉格朗日函数
    L=f+k*g
    #求导
    dx = diff(L, x)   # 对x求偏导
    # print("dx=",dx)
    dy = diff(L,y)   #对y求偏导
    # print("dy=",dy)
    # dz = diff(L,z)   #对z求偏导
    # print("dz=",dz)
    dk = diff(L,k)   #对k求偏导
    # print("dk=",dk)
    #求出个变量解
    # m = solve([dx,dy,dz,dk],[x,y,z,k])   
    # m = solve([dx,dy,dz,dk],x,y,z,k, dict=True)   
    m = solve([dx,dy,dk],x,y,k, dict=True)   
    # m = nonlinsolve([dx,dy,dz,dk],[x,y,z,k])    
    print(m)
   

# Lagrange(Fxy, g, x, y, k)