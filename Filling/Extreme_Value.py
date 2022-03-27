# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal   #滤波等

# xxx = np.arange(0, 1000)
# yyy = np.sin(xxx*np.pi/180)

# z1 = np.polyfit(xxx, yyy, 2) # 用7次多项式拟合
# p1 = np.poly1d(z1) #多项式系数
# print(p1) # 在屏幕上打印拟合多项式
# yvals=p1(xxx) 

# x1*x1-x1*x2+x3
import math
import numpy as np
import random

DNA_SIZE = 1
POP_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.015
N_GENERATIONS = 200
X_BOUND = [3.0, 5.0]  # x1
Y_BOUND = [2.1, 6.7]  # x2
Z_BOUND = [1.2, 9.6]  # x3


sum=0#测试变量

def elements_sum():
    final = (2 * x + 3 * y + 2 * z) / (x + y + z)



def F(x, y, z):
    # val = ((((2 * x + 3 * y + 2 * z) / (x + y + z) - 2) * ((2 * x + 3 * y + 2 * z) / (x + y + z) - 2)) + (((3 * x + 3 * y + 2 * z) / (x + y + z) - 2) * ((3 * x + 3 * y + 2 * z) / (x + y + z) - 2))) / 2
    # val = x * x - x * y + z +y
    val = x * x - x * y + y
    # print(val.shape) #(100,) 100 <class 'numpy.ndarray'>
    return val


def get_fitness(pop):
    x, y, z = translateDNA(pop)
    pred = F(x, y, z)
    return pred


def translateDNA(pop):  # pop表示种群矩阵, 一行表示一个二进制编码表示的DNA, 矩阵的行数为种群数目
    # x_pop = pop[:,0:DNA_SIZE]#这样的写法shape 是(3, 1) 3行1列 ndim维度是2(行, 列 矩阵 )
    # 也可以认为是二维数组, 有3行, 每行有1个元素 size为3 [[3.18796615]\n [3.32110516]\n [4.34665405]]
    '''因为这样写x_pop = pop[:, 0:DNA_SIZE] shape是(3,1)是二维数组, 所以会报"对象太深, 无法容纳所需的数组"的错误, 
    第一种解决方法是进行reshape, 比如reshape(3,)即变成了一维数组, 元素个数是3个, 即语法是x_pop=pop[:,0:DNA_SIZE].reshape(POP_SIZE,)
    这时x_pop就变为[4.96893731 3.24515899 3.51500566] 一维数组

    第二种方法是在矩阵(二维数组)pop中直接选择某一列元素, 比如  pop[:, 0],表示选择pop第0列所有的元素

    '''

    x_pop = pop[:,0]  # 取前DNA_SIZE个列表示x 这样的写法shape是(3,) ndim维度是1 一维数组 , 数组元素有3个 size为3 [4.28040552 3.25412449 4.61336022]
    # print(x_pop.shape)
    y_pop = pop[:, 1]  # 取中间DNA_SIZE个列表示y
    z_pop = pop[:, 2]  # 取后DNA_SIZE个列表示z
    # print(x_pop)

    return x_pop, y_pop, z_pop


def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 3)  # 随机产生一个实数, 代表要变异基因的位置
        if mutate_point == 0:
            child[mutate_point] = np.random.uniform(3.0, 5.0)
        elif mutate_point == 1:
            child[mutate_point] = np.random.uniform(2.1, 6.7)
        else:
            child[mutate_point] = np.random.uniform(1.2, 9.6)


def crossover_and_mutation(pop, CROSSOVER_RATE=0.015):
    new_pop = []
    for i in range(POP_SIZE//2-20):
        fatherpoint = np.random.randint(low=0, high=POP_SIZE)
        child=pop[fatherpoint]
        motherpoint = np.random.randint(low=0, high=POP_SIZE)
        cross_points = np.random.randint(low=0, high=DNA_SIZE * 3)  # 随机产生交叉的点
        child[cross_points] = pop[motherpoint][cross_points]
        new_pop.append(child)


    for i in range(20):
        fatherpoint = np.random.randint(low=0, high=POP_SIZE)
        child = pop[fatherpoint]
        mutation(child, MUTATION_RATE=1)
        new_pop.append(child)

    # for father in pop:  # 遍历种群中的每一个个体, 将该个体作为父亲
    #     child = father  # 孩子先得到父亲的全部基因
    #     if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉, 而是以一定的概率发生交叉
    #         # mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体, 并将该个体作为母亲
    #
    #         mother = pop[np.random.randint(POP_SIZE)]
    #
    #
    #         cross_points = np.random.randint(low=0, high=DNA_SIZE * 3)  # 随机产生交叉的点
    #         # child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
    #         child[cross_points] = mother[cross_points]
    #     else:
    #         mutation(child, MUTATION_RATE)  # mutation(child,MUTATION_RATE)每个后代有一定的机率发生变异
    #     new_pop.append(child)

    return new_pop


def getbest(pop, fitness):
    best_indiv = []
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    pop_copy_max = pop[max_fitness_index]
    best_indiv.append(pop_copy_max)
    # print(best_indiv)
    return max_fitness_index
def getworst(pop,fitness):
    worst_indiv = []
    fitness = get_fitness(pop)
    min_fitness_index = np.argmin(fitness)
    pop_copy_min = pop[min_fitness_index]
    worst_indiv.append(pop_copy_min)
    return min_fitness_index


def choicebyyang(arr, size, replace, p):
    for i in range(size):

        aa = np.random.rand()
        sum = 0
        for j in range(size):
            sum = sum + p[j]  # 累加概率
            if sum >= aa:
                break

        arr[i] = j
    return arr


def select(pop, fitness):  # nature selection wrt pop's fitness
    # fitnew=fitness.copy() #深拷贝
    fitnew = fitness
    # print('fitness',id(fitness))# 赋值操作在Python里是浅拷贝, 两个变量地址一样
    # print('fitnew',id(fitnew))
    # print(id(fitnew)==id(fitness)) #True
    fitnew = fitnew + 1e-3 - np.min(fitnew)
    p = (fitnew) / (fitnew.sum())
    # print(np.arange(POP_SIZE))  #产生的是一维数组[0 1 2 ...100]
    # idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=p)
    # 从0~POP_SIZE这个序列里随机取样 如果[pop_size]是一维数组, 就表示从这个一维数组中随机采样,采size个, 上面这行是全采用
    idx = choicebyyang(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=p) #一维数组
    bestindex = getbest(pop, fitness) #最优个体下标
    worstindex=getworst(pop,fitness) #最差个体下标


    # new_idx=[bestindex,worstindex] #<class 'list'>: [0, 1]
    # new_idx=[idx]+[bestindex,worstindex] #<class 'list'>: [array([0, 0, 0]), 0, 2]
    # new_idx = list(idx) + [bestindex, worstindex] #<class 'list'>: [0, 1, 1, 1, 2]
    new_idx=list(idx) #<class 'list'>: [2, 1, 2]
    half_pop_idx=new_idx[:POP_SIZE//2-2]
    half_pop_idx.append(bestindex)
    half_pop_idx.append(worstindex)

    half_pop_idx2=new_idx[POP_SIZE//2:]


    return pop[half_pop_idx],pop[half_pop_idx2] # 尽量选择适应值高的函数值的个体


'''
如果POP_SIZE=3,即种群个数是3, 则从交叉, 变异后的种群中, 选择3个适应值高 pop[idx]=[2 0 0]的新个体去
更新pop种群, 之后再进行不断的迭代, 直到达到迭代次数终止。
'''


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmin(fitness)
    # max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y, z = translateDNA(pop)

    print("最优的基因型：", pop[max_fitness_index])

    print("(x, y, z):", (x[max_fitness_index], y[max_fitness_index], z[max_fitness_index]))


if __name__ == "__main__":
    # pop1 = np.random.uniform(3.0, 5.0, size=(POP_SIZE, DNA_SIZE))  # matrix (POP_SIZE, DNA_SIZE)
    # pop2 = np.random.uniform(2.1, 6.7, size=(POP_SIZE, DNA_SIZE))
    # pop3 = np.random.uniform(1.2, 9.6, size=(POP_SIZE, DNA_SIZE))
    #
    # # print(type(pop1))# (100,1) 维度是2(行列 矩阵) <class 'numpy.ndarray'>
    # # pop={pop1,pop2,pop3}
    # pop = np.hstack((pop1, pop2, pop3)) #水平拼接
    # print(pop)
    '''
    [[3.44603448 4.51707625 7.90178727]
     [4.57616299 5.11309286 4.86911781]
     [3.24273815 2.9253602  4.45149325]
     ...
     [4.39321276 3.1657492  5.16654786]]
    '''
    # print(type(pop)) #<class 'numpy.ndarray'> n维数组
    # print(pop.shape) #(100,3) 矩阵有 100行, 3列
    # print(pop.ndim) # 2 因为矩阵有行和列两个维度
    # print(pop.size) #300  矩阵共有300个元素
    # print(pop.dtype) #float64 矩阵元素类型是float64
# for i in range(100):#测试代码
    pop1 = np.random.uniform(3.0, 5.0, size=(POP_SIZE, DNA_SIZE))  # matrix (POP_SIZE, DNA_SIZE)
    pop2 = np.random.uniform(2.1, 6.7, size=(POP_SIZE, DNA_SIZE))
    pop3 = np.random.uniform(1.2, 9.6, size=(POP_SIZE, DNA_SIZE))

    # print(type(pop1))# (100,1) 维度是2(行列 矩阵) <class 'numpy.ndarray'>
    # pop={pop1,pop2,pop3}
    pop = np.hstack((pop1, pop2, pop3)) #水平拼接
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y, z = translateDNA(pop)  # 这句代码, 我觉得没啥作用
        # print(x) #(100,) [4.82264692 4.04610252 4.92107325 4.49556859 3.1322498  3.60757363...] 一维数组100个数据
        # pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # print(pop.dtype)#<class 'numpy.ndarray'> (100, 3) 2 300 float64
        fitness = get_fitness(pop)

        # print(fitness) #<class 'numpy.ndarray'> (100,) 一维数组 100
        pop_half,pop_half2 = select(pop, fitness)  # 选择生成新的种群 50行3列

        pop_half3=np.array(crossover_and_mutation(pop, CROSSOVER_RATE)) #50行3列

        pop=np.vstack((pop_half,pop_half3)) #纵向(上下拼接)拼接 100行 3列




    print_info(pop)

    #测试 运行100次 的最大值平均值代码
#     i=i+1
#     print(i)
#     fitness = get_fitness(pop)
#     max_fitness_index = np.argmax(fitness)
#     sum=sum+fitness[max_fitness_index]
# print(sum/100.0)

print(np.hstack((np.random.uniform(3.0, 5.0, size=(2, 1)), np.random.uniform(1.0, 2.0, size=(2, 1)))))

