from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 构建模型
clf = DecisionTreeClassifier() # 决策树分类器实例化
clf.fit(x_train, y_train) # 训练数据，决策树分类器会根据训练数据的特征和目标变量来构建决策树模型，从而实现对新数据的分类

# 在测试集上进行预测
y_pred = clf.predict(x_test)

# 模型评估
# ------------ 准确率 ------------ #
acc = clf.score(x_test, y_test)
print("Accuracy: ", acc)
# ------------ 精确率、召回率、F1值 ------------ #
report = classification_report(y_test, y_pred)
print(report)
# ------------ 混淆矩阵 ------------ #
cm = confusion_matrix(y_test, y_pred)
print(cm)

train_sizes, train_scores, test_scores = learning_curve( # learning_curve函数可以用于生成学习曲线，帮助评估模型在不同训练集大小下的性能
    clf, # 训练好的分类器
    iris.data, # 训练数据
    iris.target, # 训练标签
    cv = 5, # 交叉验证的折数，这里指将原始数据集划分为5个互不重叠的子集，其中一个子集作为验证集，其余的4个子集作为训练集
    n_jobs = -1, # 指定使用的CPU数量，-1表示使用所有可用的CPU
    train_sizes = np.linspace(0.1, 1.0, 10), # 指定训练集的大小，这里使用了10个不同的训练集大小，从10%到100%
    scoring = 'accuracy' # 指定评估指标，这里使用了准确率
    )
# 运行完这段代码后，会得到一个包含训练集大小、训练集准确率和交叉验证集准确率的元组 (train_sizes, train_scores, test_scores)

# 计算平均值和标准差
train_mean = np.mean(train_scores, axis=1) # 训练集准确率均值
train_std = np.std(train_scores, axis=1) # 训练集准确率标准差
test_mean = np.mean(test_scores, axis=1) # 测试集准确率均值
test_std = np.std(test_scores, axis=1) # 测试集准确率标准差

# 绘制学习曲线图形
plt.plot(train_sizes, train_mean, label='Training score') # 绘制训练集准确率曲线
plt.plot(train_sizes, test_mean, label='Cross-validation score') # 绘制测试集集准确率曲线

# 绘制标准差区域
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1) # 绘制训练集准确率标准差区域
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, alpha=0.1) # 绘制测试集准确率标准差区域

# 添加图例和标签
plt.legend() # 添加图例
plt.xlabel('Number of training samples') # 设置x轴标签
plt.ylabel('Accuracy') # 设置y轴标签
plt.show()

# 预测测试集中每个样本的概率分布
y_score = clf.predict_proba(x_test)

# 计算每个类别的ROC曲线和AUC值
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制每个类别的ROC曲线
plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
