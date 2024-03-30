from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import IrisDataLoad
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import precision_score
from sklearn import tree

if __name__ == '__main__':
    choice = "DecisionTree"
    # choice = "RandomForest"

    # 1.加载数据集
    myIrisDataLoader = IrisDataLoad.MyIrisDataLoader()  # 需要加上()，否则会报错
    X, y, attributes, labels = myIrisDataLoader.my_load_iris_data()
    print(X)  # 150行4列
    print(y)  # 150行1列
    print(attributes)  # 四种，分别是sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
    print(labels)  # 三种，分别是setosa, versicolor, virginica

    # 打印数据集信息
    # myIrisDataLoader.print_iris_data_info()

    # 中值滤波和高斯滤波
    X = medfilt(X, kernel_size=3)
    X = gaussian_filter(X, sigma=1)

    # 2.划分数据集
    # 分割数据集为训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

    # 打印X_train和y_train的形状
    print("X_train and y_train shape:")
    print(X_train.shape)
    print(y_train.shape)

    # 3.训练模型
    # 创建决策树分类器
    clf = None
    if choice == "DecisionTree":
        clf = DecisionTreeClassifier()
    elif choice == "RandomForest":
        clf = RandomForestClassifier(n_estimators=4)
    else:
        print("Wrong choice or no choice set!")
        exit(0)

    # 训练模型
    clf.fit(X_train, y_train)

    # 在验证集上进行预测
    y_pred = clf.predict(X_test)

    # 4.评估模型
    # 计算F1-score
    # macro和micro的含义是
    # macro：对每个类别的f1-score进行算术平均
    # micro：对每个类别的f1-score进行算术平均
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    print("F1-score(macro):", f1_macro)
    print("F1-score(micro):", f1_micro)

    classfication_report = classification_report(y_test, y_pred, target_names=labels)
    print("Classification report:")
    print(classfication_report)

    # 准确度计算precision
    # 计算决策树模型的精确度
    precision = precision_score(y_test, y_pred, average='micro')

    # 5.可视化
    if choice == "DecisionTree":
        # 一、使用tree.plot_tree()方法可视化决策树
        # plt.figure(dpi=200)
        # tree.plot_tree(clf, feature_names=attributes, class_names=labels)
        # plt.show()

        # 二、使用export_graphviz()方法可视化决策树
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=attributes,
            class_names=labels,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render("iris" + "_" + choice)
    elif choice == "RandomForest":
        dot_data_rf = export_graphviz(
            clf.estimators_[0],
            out_file=None,
            feature_names=attributes,
            class_names=labels,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph_rf = graphviz.Source(dot_data_rf)
        graph_rf.render("iris" + "_" + choice)

