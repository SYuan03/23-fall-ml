import cv2
import PicRead
import ConvolutionLayer
import FullConnectLayer
import PoolingLayer
import numpy as np
import datetime
import sys
import argparse
parser = argparse.ArgumentParser(description='CNN HandWriting Recognition')
parser.add_argument('-tf', '--train_file', type=str, help='File path of picture',
                    default='./Resources/train-images.idx3-ubyte', dest='filepath')
parser.add_argument('-tl', '--train_label', type=str, help='Label file path',
                    default='./Resources/train-labels.idx1-ubyte', dest='labelpath')
parser.add_argument('-cs', '--convolution_core_size', type=int, help='Convolution conv core size', dest='convsize', default=3)
parser.add_argument('-cst', '--convolution_stride', type=int, help='Convolution layer stride', dest='convstride', default=1)
parser.add_argument('-c', '--channels', type=int, dest='channels', help='KMeans classify to how many class', default=3)
parser.add_argument('-ps', '--pool_size', type=int, dest='poolsize', default=2, help='Pooling layer size')
parser.add_argument('-pst', '--pool_stride', type=int, dest='poolstride', default=2, help='Pooling layer stride')
parser.add_argument('-t', '--types', type=int, dest='types', default=10, help='Final recognition types')
parser.add_argument('-TF', '--test_file', type=str, help='Test file path of picture',
                    default='./Resources/t10k-images.idx3-ubyte', dest='test_filepath')
parser.add_argument('-TL', '--test_label', type=str, help='Test label path of picture',
                    default='./Resources/t10k-labels.idx1-ubyte', dest='test_label')
parser.add_argument('-time', type=int, default=3, dest='time')
parser.add_argument('-lr', '--learning-rate', type=int, default=0.0001, dest='learning_rate')
args = parser.parse_args()


class CNN:
    def __init__(self, convsize, convstride, poolsize, poolstride, image_size, channel, types):
        """
        构建卷积神经网络
        :param convsize: 卷积核大小
        :param poolsize: 池化大小
        :param image_size: 图片大小
        :param channel: 通道数（卷积核数目）
        :param types: 最终分类数目
        """
        # 定义卷积层，池化层，全连接层
        self.conv = ConvolutionLayer.Convolution(convsize, convstride, channel)
        self.pool = PoolingLayer.Pool(poolsize, poolstride)
        self.full = FullConnectLayer.FullConnect(int((image_size / (convsize - convstride) ** 2) * channel), types)

    # train
    def train(self, images, target, time, learning_rate):
        """
        进行训练
        :param images: 训练图片集合（28*28*1）
        :param target: 训练集label
        :param time: 在训练集上的训练次数
        :param learning_rate: 学习率
        """
        loss_list = []
        for i in range(time):
            loss = 0  # 此轮训练的损失
            for image_index in range(len(images)):
                print("Info : train pic " + str(image_index))
                print("Info : expect type " + str(target[image_index]))
                out = self.conv.conv(images[image_index])
                out = self.pool.max_pooling(out)
                out = self.full.full_connect(out)
                print("Info : predict type " + str(np.where(out == np.max(out))))
                # 计算损失值
                loss += -np.log(out[target[image_index]])

                # 作反馈，进行训练参数的修正
                gradient = np.zeros(10)
                gradient[target[image_index]] = -1 / out[target[image_index]]

                # 三层反馈
                gradient = self.full.full_connect_feedback(gradient, learning_rate)
                gradient = self.pool.feedback(gradient)
                gradient = self.conv.feedback(gradient, learning_rate)
            loss_list.append(loss)
            print("Info : train time " + str(i) + " loss " + str(loss))
            print("Info : conv filter " + str(self.conv.filter))
            print("Info : full connect matrix " + str(self.full.full_connect_matrix))
            print("Info : full connect offset " + str(self.full.offset))

    def do_test(self, test_img, target):
        print("Info : begin test")
        correct = 0
        second_correct = 0
        for i in range(len(test_img)):
            out = self.conv.conv(test_img[i])
            out = self.pool.max_pooling(out)
            out = self.full.full_connect(out)
            predict = np.where(out == np.max(out))[0]
            second = np.argsort(out)[-2]
            print("Info : picture " + str(i) + " expect " + str(target[i]) + " predict " + str(predict))
            if predict == target[i]:
                correct += 1
                second_correct += 1
            elif second == target[i]:
                second_correct += 1
        print("Info : total correct rate " + str(correct / len(test_img)))
        print("Info : permit two choice correct rate " + str(second_correct / len(test_img)))

def test():
    test_img = []
    test_img.append(cv2.imread('img_0.png', 0))
    target = [1]
    CNN_test = CNN(3, 2, image_size=14 * 14 * 3, channel=3, types=10)
    CNN_test.train(images=test_img, target=target, time=3, learning_rate=0.001)
    pass


if __name__ == '__main__':
    # test()
    saveStdout =sys.stdout
    print_log = open('./log.txt', "w")
    sys.stdout = print_log

    print("Info : begin time " + str(datetime.datetime.now()))

    train_data_file_path = args.filepath
    train_label_file_path = args.labelpath
    cs = args.convsize
    cst = args.convstride
    c = args.channels
    ps = args.poolsize
    pst = args.poolstride
    t = args.types
    test_data_file_path = args.test_filepath
    test_label_file_path = args.test_label
    time = args.time
    lr = args.learning_rate

    img_list, (height, width) = PicRead.read_idx3_ubyte_pixel_file(file_path=train_data_file_path)
    target_list = PicRead.read_idx3_ubyte_label_file(file_path=train_label_file_path)
    test_img_list = PicRead.read_idx3_ubyte_pixel_file(file_path=test_data_file_path)[0]
    test_target_list = PicRead.read_idx3_ubyte_label_file(file_path=test_label_file_path)

    CNN_test = CNN(cs, cst, ps, pst, height * width, c, t)
    CNN_test.train(img_list, target_list, time, lr)
    CNN_test.do_test(test_img_list, test_target_list)

    print("Info : end time " + str(datetime.datetime.now()))
    sys.stdout = saveStdout
