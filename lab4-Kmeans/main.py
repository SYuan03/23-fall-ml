import cv2
import numpy as np
import matplotlib.pyplot as plt


class MyKMeans:
    # n_clusters: 聚类中心个数， 默认为 8
    # max_iter: 最大迭代次数， 默认为 400
    # tol: 质心移动距离小于该值时，停止迭代， 默认为 0.0001
    def __init__(self, n_clusters: int = 8, max_iter: int = 400, init_gen_center_way: str = 'kmeans'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # self.tolerance = tolerance 可以考虑用，相当于一个聚类中心移动量的阈值
        self.init_gen_center_way = init_gen_center_way

        self.cluster_center_list = None
        self.sample_label_list = None
        self.sample_label_list_prev = None
        self.dist = None

    def __gen_init_k_center(self, x_train):
        n_sample = x_train.shape[0]  # 样本个数
        n_feature = x_train.shape[1]  # 特征个数，即RGB三个通道，每个样本的特征

        if self.init_gen_center_way == 'kmeans':
            # 使用Kmeans的方式(random)初始化聚类中心
            # 为了在数据范围内产生随机k个聚类中心，首先计算各特征的统计量
            # 否则可能超出范围
            f_mean = np.mean(x_train, axis=0)  # 每一列(r/g/b)的均值
            f_std = np.std(x_train, axis=0)

            print("f_mean: ", f_mean)
            print("f_std: ", f_std)

            # np.random.randn(x, y)生成x行y列的随机数，服从标准正态分布
            self.cluster_center_list = f_mean + np.random.randn(self.n_clusters, n_feature) * f_std

        # elif self.init_gen_center_way == 'kmeans++':
        #     # 使用随机的方式初始化聚类中心
        #     # 第一个质心随机选
        #     idx = np.random.randint(0, n_sample)
        #     first_random_center = [x_train[idx, :]]
        #
        #     # 按算法选出后面k-1个质心
        #     # for i in range(1, self.n_clusters):

    def fit(self, x_train):
        n_sample = x_train.shape[0]

        # 生成初始聚类中心
        self.__gen_init_k_center(x_train)

        # 初始化样本到聚类中心的距离为0
        # 样本数 * 聚类中心数，每个样本到每个聚类中心的距离
        self.dist = np.zeros((n_sample, self.n_clusters))
        # 初始化样本的标签为-1
        self.sample_label_list = np.zeros(n_sample) - 1

        epoch = 0
        # epoch 小于最大迭代次数，且聚类中心移动距离大于阈值时，继续迭代
        while epoch < self.max_iter:

            # 计算每个样本到每个聚类中心的距离
            for i in range(self.n_clusters):
                # x_train是n_sample * n_feature的矩阵，每一行是一个样本
                # axis=1,按行求范数,消去n_feature，只剩下n_sample，即每个样本到聚类中心的距离
                # dist[:, i]表示第i个聚类中心到所有样本的距离，左右都是列向量
                self.dist[:, i] = np.linalg.norm(x_train - self.cluster_center_list[i], axis=1)

            # 遍历dist每一行，根据距离分类
            for i in range(n_sample):
                # 样本对应的类别为距离最近的聚类中心
                self.sample_label_list[i] = np.argmin(self.dist[i])

            # 打印一些信息,如epoch,聚类中心
            print('epoch: ', epoch)
            print('cluster_center_list: ', self.cluster_center_list)
            print()

            # 比较分类是否发生变化，如果不变则退出
            if np.array_equal(self.sample_label_list, self.sample_label_list_prev):
                break

            # 保存下现在的标签情况
            self.sample_label_list_prev = np.copy(self.sample_label_list)

            # 计算新的聚类中心
            for i in range(self.n_clusters):
                # 按列求均值，即求每个聚类中心的新位置
                # axis=0,按列求均值，消去n_sample，只剩下n_feature，即每个聚类中心的新位置
                self.cluster_center_list[i] = np.mean(x_train[self.sample_label_list == i], axis=0, keepdims=True)
                # 注意这里可能有空簇，即某个聚类中心没有样本，这时候均值为nan，需要处理
                # 一种是考虑将其删除（不能删？删了还算kmeans吗？）
                # 一种是为其重新生成一个随机聚类中心
                # 这里采用第二种方法
                if np.isnan(self.cluster_center_list[i]).any():
                    # 为了在数据范围内产生随机k个聚类中心，首先计算各特征的统计量
                    # 否则可能超出范围
                    f_mean = np.mean(x_train, axis=0)
                    f_std = np.std(x_train, axis=0)
                    # 生成新的聚类中心：f_mean + k*f_std, 其中 k 是一堆服从标准正态分布的随机变量
                    self.cluster_center_list[i] = f_mean + np.random.randn(*f_mean.shape) * f_std

            epoch += 1

    # 输入测试集，输出预测的聚类标签
    # 图像分割用不到这个函数
    # def predict(self, x_test):
    #     n_sample = x_test.shape[0]
    #     dist_test = np.zeros((n_sample, self.n_clusters))
    #
    #     for i in range(self.n_clusters):
    #         dist_test[:, i] = np.linalg.norm(x_test - self.cluster_center_list[i], axis=1)
    #     clus_pred = np.argmin(dist_test, axis=1)
    #
    #     return clus_pred


if __name__ == '__main__':
    # 读取图片， 转换为RGB格式
    img_name = 'cyt'
    img = cv2.imread(img_name + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示图片
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    # plt.axis('off') # 不显示坐标轴
    plt.show()

    # 获取图片的高和宽
    w, h = img.shape[0], img.shape[1]

    # 打印图片的高和宽
    print('w: ', w)
    print('h: ', h)

    # img本来是w * h * 3的三维矩阵，现在转换为w * h行，3列的二维矩阵
    img = img.reshape((w * h, 3))

    n_clusters_ = 5
    max_iter_ = 200
    # 聚类
    kmeans = MyKMeans(n_clusters_, max_iter_, init_gen_center_way='kmeans')
    kmeans.fit(img)

    # 给不同类别准备的颜色，一共10种颜色
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

    # 把每个像素点的颜色换成聚类中心的颜色
    output = np.zeros((w, h, 3))
    for i_ in range(w):
        for j_ in range(h):
            output[i_, j_, :] = colors[int(kmeans.sample_label_list[i_ * h + j_])]
    output = output * 255

    # 显示图片
    plt.figure(figsize=(10, 10))    # 设置窗口大小
    plt.imshow(output.astype(np.uint8))
    # plt.axis('off') # 不显示坐标轴
    plt.show()
    # 保存图片到dogcat目录下
    cv2.imwrite(img_name + '/' + img_name + '_kmeans_' + str(n_clusters_) + '.png', output)
