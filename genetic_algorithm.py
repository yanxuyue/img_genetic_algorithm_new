import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 图片数组
pics = np.array(["./B_img/000.bmp", "./B_img/001.bmp", "./B_img/002.bmp", "./B_img/003.bmp", "./B_img/004.bmp",
                 "./B_img/005.bmp", "./B_img/006.bmp", "./B_img/007.bmp", "./B_img/008.bmp", "./B_img/009.bmp",
                 "./B_img/010.bmp", "./B_img/011.bmp", "./B_img/012.bmp", "./B_img/013.bmp", "./B_img/014.bmp",
                 "./B_img/015.bmp", "./B_img/016.bmp", "./B_img/017.bmp", "./B_img/018.bmp"])


# 存储图片左右像素矩阵
picspixel = np.zeros((19, 2), dtype=object)

for i in range(19):
    x = Image.open(f"{pics[i]}")
    y = np.array(x.convert("L"))  # 变成像素矩阵
    picspixel[i, 0] = y[:, 0]
    picspixel[i, 1] = y[:, -1]

# 计算并存储不匹配度
diftable = np.zeros((19, 19))
for i in range(19):
    for j in range(19):
        diftable[i, j] = np.sum(abs(picspixel[i, 1] - picspixel[j, 0]))

# 初始种群,允许重复的排列
n = 100  # 族群内的个数,用偶数
first_generation = np.zeros((n, 19), dtype=int)
for i in range(n):
    first_generation[i] = np.random.permutation(19)

# 适配度
fitness = np.zeros(n)
for i in range(n):
    sums = 0
    for j in range(18):
        sums += diftable[first_generation[i][j], first_generation[i][j+1]]
    fitness[i] = 1/(sums+1e-8)


# 初始化
iteration = 0  # 迭代进程
iterations = 800  # 最大迭代次数
difmean = np.zeros(iterations)  # 每代平均代价
difbest = np.zeros(iterations)  # 每代最优代价
pathbest = np.zeros((iterations, 19))  # 每代最优排序
bestpath = np.zeros(19)  # 全局最优解
mutation_p = 0.01  # 变异概率,小于才变
span_num = 40  # 逆转次数

while iteration < iterations:
    sons = np.zeros((n, 19), dtype=int)  # 暂时存储每一代
    p = (fitness / sum(fitness)).cumsum()  # 选择概率
    # 产生子代
    for i in range(n//2):
        # 选出两个父代
        a = p - np.random.rand()
        f_1 = first_generation[(np.where(a >= 0)[0])[0]]
        b = p - np.random.rand()
        f_2 = first_generation[(np.where(b >= 0)[0])[0]]
        # 顺序交叉
        j, g = np.sort(np.random.choice(19, 2, replace=False))
        l_1 = f_1[j:g]
        l_2 = f_2[j:g]  # 要交换的两个切片
        l_3 = [x for x in list(f_1[g:]) + list(f_1[:g]) if x not in l_2]
        l_4 = [x for x in list(f_2[g:]) + list(f_2[:g]) if x not in l_1]
        # 生成子列
        child_1 = np.full(19, -1, dtype=int)
        child_2 = np.full(19, -1, dtype=int)
        child_1[:j] = l_4[:j]
        child_1[j:g] = l_1[:]
        child_1[g:] = l_4[j:]
        child_2[:j] = l_3[:j]
        child_2[j:g] = l_2[:]
        child_2[g:] = l_3[j:]
        # 存储子列
        sons[2*i] = child_1
        sons[2*i+1] = child_2
    # --------精英保留---------
    elite_indices = np.argsort(fitness)[-2:]  # 上一代适应度最高的两个
    sons[-2] = first_generation[elite_indices[0]].copy()
    sons[-1] = first_generation[elite_indices[1]].copy()
    # -------精英保留结束-------
    # 变异
    for i in range(n-2):
        x = np.random.rand()
        if x < mutation_p:
            # 进行变异
            j, g = np.sort(np.random.choice(19, 2, replace=False))
            sons[i][j], sons[i][g] = sons[i][g], sons[i][j]

    # 计算每个子代的适配度
    son_fitness = np.zeros(n)
    for i in range(n):
        cum = 0
        for j in range(18):
            cum += diftable[sons[i][j], sons[i][j+1]]
        son_fitness[i] = 1/(cum+1e-8)
    # 进化逆转
    for w in range(span_num):
        son_span = np.zeros((n, 19), dtype=int)
        for i in range(n):
            k_1, k_2 = np.sort(np.random.choice(19, 2, replace=False))
            son_span[i, :k_1] = sons[i, :k_1]
            son_span[i, k_1:k_2] = sons[i, k_1:k_2][::-1]
            son_span[i, k_2:] = sons[i, k_2:]
        # 计算逆转后的子代适配度
        son_fitness_span = np.zeros(n)
        for i in range(n):
            cums = 0
            for j in range(18):
                cums += diftable[son_span[i][j], son_span[i][j+1]]
            son_fitness_span[i] = 1/(cums+1e-8)
        # 留下优质的逆转
        for i in range(n):
            if son_fitness_span[i] > son_fitness[i]:
                sons[i] = son_span[i]
                son_fitness[i] = son_fitness_span[i]  # 更新适应度
    # 临时代价矩阵
    temdif = np.zeros(n)
    for i in range(n):
        c = 0
        for j in range(18):
            c += diftable[sons[i][j], sons[i][j+1]]
        temdif[i] = c
    # 每代平均代价
    difmean[iteration] = np.mean(temdif)
    # 每代最优代价
    difbest[iteration] = np.min(temdif)
    # 每代最优解
    pathbest[iteration] = sons[np.argmin(temdif)]
    # 更新数据
    first_generation = sons.copy()
    fitness = son_fitness.copy()

    iteration += 1

# 全局最优解
bestpath = pathbest[np.argmin(difbest)].astype(int)

# 展示平均代价和每代最优解
fig, ax = plt.subplots(2, 1, figsize=(13, 11))

ax[0].plot(np.arange(iterations), difmean, 'k')
ax[0].set_title('Mean Difference')
ax[0].set_xlabel('Iteration')

ax[1].plot(np.arange(iteration), difbest, 'k')
ax[1].set_title('Best Difference')
ax[1].set_xlabel('Iteration')

fig.savefig("dif_02.png")
plt.show()

# 拼接纸片(横向)
image = [Image.open(f"./B_img/{int(x):03d}.bmp") for x in bestpath]
total_width = sum([im.width for im in image])
max_height = max([im.height for im in image])
img = Image.new('L', (total_width, max_height))
offset_x = 0
for im in image:
    img.paste(im, (offset_x, 0))
    offset_x += im.width
img.save("merged_result02.bmp")
img.show()

