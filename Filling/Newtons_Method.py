# import numpy as np
# import matplotlib.pyplot as plt

# print(np.array([[2,-1],[-1,2]]))
# #牛顿法求解f = 60-10*x1-4*x2+x1**2+2*x2**2-x1*x2的极值
# #原函数
# #建立点的列表并关联起来
# X1=np.arange(-50,50,0.1)
# X2=np.arange(-50,50,0.1)
# [x1,x2]=np.meshgrid(X1,X2)
# # f = 60-10*x1-4*x2+x1**2+2*x2**2-x1*x2
# f = (x1 - 10)**2 + (x2 - 20)**2
# plt.contour(x1,x2,f,20) # 画出函数的20条轮廓线

# #求梯度
# def jacobian(x):
#     return np.array([-10+2*x[0]-x[1],-4+2*x[1]-x[0]])

# #求海森矩阵
# def hessian(x):
#     return np.array([[2,-1],[-1,2]])

# #牛顿法程序
# def newton(x0):
#     W=np.zeros((2,10**2))
#     i = 1
#     imax = 100
#     W[:,0] = x0 
#     x = x0
#     delta = 1
#     alpha = 1
    
#     #迭代条件，当迭代次数少于100次并且精度大于0.1时进行循环
#     while i<imax and delta>0.1:
#         #将海森矩阵的逆与梯度相乘
#         p = -np.dot(np.linalg.inv(hessian(x)),jacobian(x))
#         x0 = x
#         x = x + alpha*p
#         W[:,i] = x
#         delta = sum((x-x0))
#         print('第',i,'次迭代结果:')
#         print(x,'\n')
#         i=i+1
#     W=W[:,0:i]  # 记录迭代点
#     return W

# #初始点
# x0 = np.array([-40,40])
# W = newton(x0)
    
# plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
# plt.show()



import numpy as np
from sympy import *
import numpy.matlib 


# syms x y z
x, y, z = symbols('x y z')
# Fxyz = (x-1)**3+(y-5)**2+(z-100)**2
# Fxyz = (((25 * x + 37 * y + 30 * z) / (x + y + z) - 30)** 2 + ((30 * x + 27 * y + 20 * z) / (x + y + z) - 20)** 2) / 2
# Fxyz = 4*(x + 1)**2 + 2*(y - 2)**2 + x + y + 10
Fxyz = x**2+2*y**2+3*z**2+2*x+4*y-6*z 

f1 = diff(Fxyz,x)
f2 = diff(Fxyz,y)
f3 = diff(Fxyz,z)

# F = np.mat([f1,f2,f3]).reshape(3,1) # 列表转矩阵
# print(F)
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
x0 = np.mat([1,1,1]).reshape(3,1)
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
Fxyz.evalf(subs = {'x': x1[0,0], 'y': x1[1,0], 'z':x1[2,0]})
# Fxyz.evalf(subs = {'x': 1, 'y': 1, 'z':1})



# 二元函数
# f1 = diff(Fxyz,x)
# f2 = diff(Fxyz,y)

# F = Matrix([f1,f2])
# def hessian(*para):
#     all = []
#     for f_i in para[0]:
#         one = []
#         for o_i in para[1]:
#             one.append(diff(f_i, o_i))
#         all.append(one)
#         # sp.diff(f,o).evalf(subs ={'x2':6})
#     # return np.mat(all).reshape(3,3)
#     return Matrix(all)

# # J_be=[[sp.diff(f1,x), sp.diff(f1, y), sp.diff(f1,z)],
# #    [sp.diff(f2,x), sp.diff(f2, y), sp.diff(f2,z)],
# #    [sp.diff(f3,x), sp.diff(f3, y), sp.diff(f3,z)]]

# # print(J_be)
# # J = np.mat(J_be).reshape(3,3)
# J = hessian((f1, f2), (x, y))

# n=1
# x0=np.mat([1,1]).reshape(2,1)
# # mar = J.subs(dict(x=x0[0,0], y=x0[1,0], z=x0[2,0]))
# # print(mar, mar.inv())

# E=1
# while n < 50 and E > 1e-4:
#     dic = dict(x=x0[0,0], y=x0[1,0])
#     x1=x0-J.subs(dic).inv()*F.subs(dic)
#     step1 = (diag(x1[0,0], x1[1, 0])).inv() * abs(x1-x0)
#     # print('step1' , step1)
#     # E=max(abs(x1-x0)/x1)
#     E=max(step1)
#     x0=x1
#     n=n+1


# print(x1, n)
# # x=x1(1)
# # y=x1(2)
# # z=x1(3)
# # eval(F)
# Fxyz.evalf(subs = {'x': x1[0,0], 'y': x1[1,0]})
# # Fxyz.evalf(subs = {'x': 0, 'y': 0})
