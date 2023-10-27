import cv2
import numpy as np

import ActivatingFunction


class Convolution:
    filter_size = None
    filter_num = None  # 卷积核的数目是用来确定提取多少个特征，比如一个卷积核用来提取嘴，另一个用来提取手
    filter = None
    bias = None  # TODO
    padding_size = None
    img = None
    img_shape = None
    padding = None
    strider = 1

    def __init__(self, filter_size, strider=1, filter_num=1):
        """
        卷积层初始化
        :param filter_size: 卷积核大小
        :param strider: 卷积步长
        :param filter_num: 卷积核的数目 / 通道数
        """
        self.filter_size = filter_size
        self.padding_size = filter_size // 2
        self.stride = strider
        self.filter_num = filter_num
        self.filter = np.random.rand(filter_num, filter_size, filter_size)
        print("Info : convolution layer init")
        print("Info : filter size " + str(filter_size))
        print("Info : filter stride " + str(strider))
        print("Info : filter num / channel " + str(filter_num))

    def conv(self, img):
        """
        进行卷积
        :param img: 传入图片 28 * 28 * 1
        :return: 卷积结果
        """
        print("Info : begin convolution")
        height, width = img.shape
        pixel = np.asarray(img)
        self.img_shape = pixel.shape
        self.img = pixel
        pixel = np.pad(pixel, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size)),
                       mode='constant', constant_values=(0, 0))
        self.padding = pixel
        result = []
        # 进行卷积
        # 将一个个视野与卷积核进行直接相乘
        for i in range(0, (height + 2 * self.padding_size - self.filter_size) // self.stride + 1, self.stride):
            result.append([])
            for j in range(0, (width + 2 * self.padding_size - self.filter_size) // self.stride + 1, self.stride):
                result[i // self.stride].append(
                            np.sum(self.filter * (pixel[i: i + self.filter_size, j: j + self.filter_size]), axis=(1, 2)))
        # print("Info : img after convolution " + str(result))
        return result

    def feedback(self, feedback_info, learning_rate):
        """
        卷积层反馈
        :param feedback_info: 池化层反馈结果
        :param learning_rate: 学习率
        """
        print("Info : begin conv feedback")
        # print("Info : feedback " + str(feedback_info))
        # feedback.shape 在 stride 为 1 的情况下为 28 * 28 * 3
        # 与卷积结果相同
        filters = np.zeros(self.filter.shape)
        for i in range(0, (self.img_shape[0] + 2 * self.padding_size - self.filter_size) // self.stride + 1, self.stride):
            for j in range(0, (self.img_shape[1] + 2 * self.padding_size - self.filter_size) // self.stride + 1, self.stride):
                for k in range(self.filter_num):
                    # 卷积层反馈，将卷积结果的每一个误差反馈到卷积核的卷积参数上
                    filters[k] += feedback_info[i, j, k] * self.padding[i: i + self.filter_size, j: j + self.filter_size]
        self.filter -= learning_rate * filters

def test():
    test_img = cv2.imread('img_0.png', 0)
    print(test_img)
    print(test_img.shape)
    r = Convolution(3, 1, 3)
    return r.conv(test_img)


