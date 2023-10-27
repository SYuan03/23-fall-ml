import numpy as np

import ConvolutionLayer


class Pool:
    size = 2
    stride = 2
    input_img = None
    input_img_shape =None

    def __init__(self, stride=2, size=2):
        """
        池化层初始化
        :param stride: 池化步长
        :param size: 池化大小
        """
        self.stride = stride
        self.size = size  # 通常与步长相同
        print("Info : pooling layer init")
        print("Info : pooling size " + str(self.size) + " * " + str(self.size))
        print("Info : pooling stride " + str(self.stride) + " usually the same as pooling size")

    def max_pooling(self, conv_img):
        """
        采用最大池化
        :param conv_img: 卷积结果
        :return: 池化结果
        """
        print("Info : begin max pooling for conv img")
        height, width, depth = np.asarray(conv_img).shape
        # 最大池化
        conv_img = np.asarray(conv_img)
        self.input_img = conv_img
        self.input_img_shape = self.input_img.shape
        result = []
        for i in range(0, height, self.stride):
            result.append([])
            for j in range(0, width, self.stride):
                result[i // self.stride].append([])
                for k in range(depth):
                    sub = conv_img[i: i + self.size, j: j + self.size, k: k + 1]
                    result[i // self.stride][j // self.stride].append(np.max(sub))
        # print("Info : img pooled " + str(result))
        # 到这边是14 * 14 * 3 的一个矩阵
        return result

    def feedback(self, feedback_info):
        """
        池化层反馈
        :param feedback_info: 全连接层反馈结果
        :return: 池化层反馈结果
        """
        print("Info : begin pooling layer feedback")
        # print(feedback_info)
        # print(np.asarray(feedback_info).shape)
        # input_nodes = 28 * 28 * 3
        input_nodes = np.zeros(self.input_img_shape)
        height, width, depth = self.input_img_shape
        for i in range(0, height, self.stride):
            for j in range(0, width, self.stride):
                for k in range(depth):
                    sub = self.input_img[i: i + self.size, j: j + self.size, k: k + 1]
                    a = np.max(sub)
                    if self.input_img[i][j][k] == a:
                        input_nodes[i][j][k] = feedback_info[i // self.stride][j // self.stride][k]
        # 将14 * 14 * 3 的池化层的误差回退到卷积层层面
        return input_nodes

def test():
    img_conv = ConvolutionLayer.test()
    poolLayer = Pool()
    pool_img = poolLayer.max_pooling(img_conv)
    return pool_img
    pass

