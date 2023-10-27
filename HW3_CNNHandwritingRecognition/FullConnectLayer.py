import numpy as np

import ActivatingFunction
import PoolingLayer


class FullConnect:
    full_connect_matrix = None
    offset = None
    gap_matrix = None
    types = None
    input_img = None
    input_img_shape = None
    out = None
    def __init__(self, pic_size, types):
        """
        全连接层初始化
        :param pic_size: 传入的池化后的矩阵大小
        :param types: 所有的类别数目
        """
        self.full_connect_matrix = np.random.rand(pic_size, types) / pic_size
        self.offset = np.zeros(types)
        print(self.full_connect_matrix)
        print("Info : full connect layer init")
        print("Info : full connect matrix size " + str(self.full_connect_matrix.shape))
        print("Info : full connect offset size " + str(self.offset.shape))
        pass
    """
    作全连接，根据前面的Convolution和Pooling层计算之后，28*28的灰度矩阵变成了14*14*3的矩阵
    Height * Width 变成了 (Height / (Conv.stride * Pool.stride))  * (Width / (Conv.stride * Pool.stride)) * Conv.filter_num
    """
    def full_connect(self, pool_img):
        """
        进行全连接
        :param pool_img: 池化结果
        :return: 分类预测，为一个一维的，大小为类别数的矩阵
        """
        print("Info : begin full connect")
        img_array = np.asarray(pool_img)
        self.input_img = img_array.flatten()
        self.input_img_shape = img_array.shape
        height, width, depth = img_array.shape

        img_flatten = img_array.flatten()
        result = np.dot(img_flatten, self.full_connect_matrix) + self.offset
        self.last_total = result

        out = np.exp(result)
        # print("Info : full connect result " + str(result))
        return out / np.sum(out, axis=0)

    def full_connect_feedback(self, gradients, learn_rate):
        """
        全连接层反向传播
        :param gradients: 预测差
        :param learn_rate: 学习率
        :return:
        """
        print("Info : begin full connect feedback")
        # print("Info : gradients " + str(gradients))
        for i, gradient in enumerate(gradients):  # i为下标，gradient为具体的值
            # print(gradients)
            if gradient != 0:
                exps = np.exp(self.last_total)
                s = np.sum(exps)

                out_back = -exps[i] * exps / (s ** 2)
                # 反馈
                out_back[i] = exps[i] * (s - exps[i]) / (s ** 2)
                # 将反馈数值和概率做乘积，得到结果权重1
                out_back = gradient * out_back

                # 最后的输出与结果反馈的权重做点乘，获得权重的偏置
                weight_back = self.input_img[np.newaxis].T @ out_back[np.newaxis]
                inputs_back = np.dot(self.full_connect_matrix, out_back)
                # print("Info : full connect matrix before " + str(self.full_connect_matrix))
                print("Info : predict before " + str(np.dot(self.input_img, self.full_connect_matrix) + self.offset))
                self.full_connect_matrix -= learn_rate * weight_back
                # print("Info : full connect matrix after " + str(self.full_connect_matrix))
                print("Info : predict after " + str(np.dot(self.input_img, self.full_connect_matrix) + self.offset))
                self.offset -= learn_rate * out_back

                # 将矩阵从 1d 转为 3d
                # 588 to 14x14x3
        return inputs_back.reshape(self.input_img_shape)


def test():
    pool_img = PoolingLayer.test()
    full_con = FullConnect(14 * 14 * 3, 10)
    full_con.full_connect(pool_img)

