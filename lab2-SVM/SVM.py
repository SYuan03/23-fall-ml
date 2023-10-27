import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# 生成SVM数据
np.random.seed(0)

# np.r_按row来组合array，np.c_按column来组合array
# X是一个40行2列的矩阵，前20行是均值为-2的二维正态分布，后20行是均值为2的二维正态分布
X = np.r_[np.random.randn(20, 2) - [2, 2],
          np.random.randn(20, 2) + [2, 2]]
Y = [-1] * 20 + [1] * 20

# 创建SVM模型
clf = svm.SVC(kernel='linear')  # 创建线性SVM分类器
clf.fit(X, Y)  # 拟合模型



# 画出超平面
x_min = X[:, 0].min() - 2
x_max = X[:, 0].max() + 2
y_min = X[:, 1].min() - 2
y_max = X[:, 1].max() + 2

# 生成网格点矩阵，以0.02为步长
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

print("xx:")
print(xx.shape)
print(xx)

print("yy:")
print(yy.shape)
print(yy)

# xx.ravel()将xx变成一维数组
# np.c_按column来组合array, 组合后的结果是将xx和yy中的元素一一对应组合成一个二维数组
print("xx.ravel:")
print(xx.ravel().shape)
print(xx.ravel())

print("np.c_[xx.ravel(), yy.ravel()]:")
print(np.c_[xx.ravel(), yy.ravel()].shape)
print(np.c_[xx.ravel(), yy.ravel()])


Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # 计算每个点到超平面的距离

print("Z1:")
print(Z.shape)
print(Z)

# 这一步是为了将Z的维度转换成xx的维度
# 意思就是比如Z是一个一维数组，xx是一个二维数组，那么Z就会被转换成二维数组
Z = Z.reshape(xx.shape)

print("Z2:")
print(Z.shape)
print(Z)

# 绘制等高线，用以可视化超平面
# contour用于绘制等高线
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])

# 绘制数据点
# 手动指定颜色，例如，蓝色和红色
colors = ['b' if label == -1 else 'r' for label in Y]
plt.scatter(X[:, 0], X[:, 1], c=colors, marker='o')

# 绘制支持向量
# 圈出支持向量，即在图上圈出支持向量点
support_vectors = clf.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolor='None', edgecolors='k')

plt.title('SVM Classifier')  # 设置图的标题
plt.savefig('./SVM.png')    # 保存图
plt.show()  # 显示图

# 输出支持向量
print("Support Vectors:")
print(support_vectors)


