#!/usr/bin/env python
# coding: utf-8
import cv2

# # KMeans 实现和应用

# ## KMeans 算法原理

# KMeans 是一种基本的动态聚类算法，核心思想是每个点都归属于它离得最近的质心代表的点。
# 
# KMeans 算法的基本步骤如下
# 
# - 随机挑选 k 个点作为初始的聚类质心
# 
# - 计算每个点到每个质心的距离，根据距离把每个点划分到离它最近的质心所属的 cluster
# 
# - 重新计算质心，如果质心无变化，则迭代结束，否则回到上一步
# 
# 在基础的 KMeans 算法的基础上，还可以在第一步选择质心的步骤上进行优化，以选择更好的初值。
# 
# KMeans 的优化版本叫做 KMeans ++，其和 KMeans 唯一的不同在于第一步选择质心。
# 
# KMeans++ 想要让初始时选择的 k 个质心之间距离尽可能远，具体的方法是:
# 
# - 首先挑出一个样本点作为第一个质心
# 
# - 计算其他样本点到第一个质心的距离 $D(x)$，计算被选择的概率 $\frac{D(x)^2}{\Sigma D(x^{(i)})^2}$，并根据轮盘法选择下一个质心
# 
# - 重复直到选出 K 个质心

# ## KMeans 算法实现

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[22]:


class KMeans:
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', max_iter: int = 300, tol: float = 0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_centers_ = None
        self.dist = None
        self.labels_ = None

    def __gen_center(self, X_train):
        n_sample, n_feature = X_train.shape

        if self.init == 'random':
            # 为了在数据范围内产生随机质心，首先计算各特征的统计量
            f_mean = np.mean(X_train, axis=0)
            f_std = np.std(X_train, axis=0)
            self.cluster_centers_ = f_mean + np.random.randn(self.n_clusters, n_feature) * f_std

        elif self.init == 'k-means++':
            # 第一个质心随机选
            idx = np.random.randint(0, n_sample)
            self.cluster_centers_ = [X_train[idx, :]]

            # 选出后面k-1个质心
            for i in range(1, self.n_clusters):
                dist = np.zeros((n_sample, len(self.cluster_centers_)))  # 各样本到质心的距离矩阵
                for cent_idx in range(len(self.cluster_centers_)):
                    dist[:, cent_idx] = np.linalg.norm(
                        X_train - self.cluster_centers_[cent_idx], axis=1)

                dist = np.min(dist, axis=1)  # 所有样本离各质心距离的最小值
                p = dist / np.sum(dist)  # 归一化后的最小距离当做概率进行下一个质心的选取，这里没有计算平方

                next_cent_idx = np.random.choice(n_sample, p=p)
                self.cluster_centers_.append(X_train[next_cent_idx])
            self.cluster_centers_ = np.array(self.cluster_centers_)

    def fit(self, X_train):
        n_sample, n_feature = X_train.shape

        self.__gen_center(X_train)
        self.dist = np.zeros((n_sample, self.n_clusters))

        cent_pre = np.zeros(self.cluster_centers_.shape)
        cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)

        epoch = 0
        from copy import deepcopy
        while epoch < self.max_iter and cent_move > self.tol:
            epoch += 1

            # 首先计算每个样本离每个质心的距离
            for i in range(self.n_clusters):
                self.dist[:, i] = np.linalg.norm(X_train - self.cluster_centers_[i], axis=1)

            # 样本对应的类别为距离最近的质心
            self.labels_ = np.argmin(self.dist, axis=1)

            cent_pre = deepcopy(self.cluster_centers_)

            # 计算每个类别下的均值坐标，更新质心
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = np.mean(X_train[self.labels_ == i], axis=0)

            cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)

    def predict(self, X_test):
        n_sample = X_test.shape[0]
        dist_test = np.zeros((n_sample, self.n_clusters))

        for i in range(self.n_clusters):
            dist_test[:, i] = np.linalg.norm(X_test - self.cluster_centers_[i], axis=1)
        clus_pred = np.argmin(dist_test, axis=1)

        return clus_pred


# 可以看到结果基本上符合我们肉眼的观察。

# ## KMeans 实现图像分割

# 下面将用 KMeans 算法实现无监督的图像分割，我这里采用的是按照颜色作为分割标准。
# 
# 具体来说，包括以下几步:
# 
# - 把每个像素映射到 rgb 空间中的点
# 
# - 做聚类
# 
# - 把结果转成可展示的图片

# In[29]:


# In[30]:


image = cv2.imread('peiqi.png')

# 以下是原始图片，我选取了我觉得不错的一幅 AI 创作的画

# In[33]:


plt.imshow(image)

# In[35]:


# resize 到合适的大小
image = cv2.resize(image, (256, 256))

# In[36]:


plt.imshow(image)

# In[39]:


image

# In[40]:


image.shape

# In[47]:


# 把每个像素看成 rgb 空间中的点，整张图片一共有 65536 个点
# 我这里是根据颜色聚类，因此在 rgb 空间里
dots = []
for i in range(256):
    for j in range(256):
        dots.append(image[i, j, :])

# In[51]:


len(dots)

# In[53]:


dots = np.array(dots)

# In[54]:


dots.shape

# In[109]:


segmentor = KMeans(n_clusters=6)
segmentor.fit(dots)
labels = segmentor.predict(dots)

# In[110]:


labels

# In[111]:


# 给不同类别准备的颜色
colors = [
    [0, 0, 1.],  # 蓝色
    [1., 0., 0.],  # 红色
    [0., 1., 0.],  # 绿色
    [1., 1., 0.],  # 黄色
    [0., 1., 1.],  # 青色
    [1., 0., 1.],  # 品红色
    [0.7, 0.2, 0.1],  # 橙色
    [0.2, 0.7, 0.1],  # 橄榄绿
    [0.9, 0.5, 0.2],  # 棕色
    [0.5, 0.5, 0.5]  # 灰色
]

# In[112]:


# 把每个点染上对应类别的颜色
segmented = np.zeros((256, 256, 3))
for i in range(256):
    for j in range(256):
        index = i * 256 + j
        segmented[i, j, :] = colors[labels[index]]

# In[113]:


# 展示染色后的图片
plt.imshow(segmented)

# 保存图片
segmented = segmented * 255
segmented = segmented.astype(np.uint8)
cv2.imwrite('segmented.png', segmented)
