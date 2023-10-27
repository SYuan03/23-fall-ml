from sklearn.datasets import load_iris


class MyIrisDataLoader:

    def __init__(self):
        self.iris = load_iris()
        # Iris数据集包含数据和标签
        self.X = self.iris.data  # 特征数据
        self.y = self.iris.target  # 目标标签

        self.attributes = self.iris.feature_names  # 属性名称
        self.labels = self.iris.target_names  # 标签名称

    def my_load_iris_data(self):
        return self.X, self.y, self.attributes, self.labels

    def print_iris_data_info(self):
        # 可以查看数据集的描述信息
        print(self.iris.DESCR)

        print(type(self.X))
        print(type(self.y))
        print(self.X.shape)
        print(self.y.shape)

        print(self.labels)
        print(self.attributes)
