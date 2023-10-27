import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import tqdm

# 最终版本代码


# Step2 定义一个卷积神经网络
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        # 卷积层1
        # 参数分别是输入通道数，输出通道数，卷积核大小
        # in_channels指明了输入的通道数，这里是3，因为输入的是RGB图像
        # out_channels指明了输出的通道数，我们将从中提取18个特征
        self.conv1 = nn.Conv2d(in_channels, in_channels * 6, 5)

        # 池化层1
        # 参数分别是池化核大小，步长
        # 这表示使用最大池化操作，池化核的大小是2x2，步长也是2。这意味着在每次池化操作中，特征图被划分为2x2的区域，
        # 然后在每个区域内选择最大值作为输出。步长为2表示池化操作会每隔2个像素进行一次操作，不会重叠。
        # 池化层用于下采样，通过减小特征图的大小，保留最显著的特征，从而减少计算复杂性。
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2
        # 这里的输入通道数是18，因为上一层的输出是18个特征图
        # 输出通道数是48，我们将从中提取48个特征
        self.conv2 = nn.Conv2d(in_channels * 6, in_channels * 16, 5)

        # 池化层2
        # 其实这里的池化层和上面的池化层1是一样的
        # 经过该层后，每张图片对应 16 * 5 * 5 * 3个特征
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层1
        # 这里的输入是16*5*5*3
        self.fc1 = nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels)

        # 全连接层2
        # 这里的输入是120，输出是84
        self.fc2 = nn.Linear(120 * in_channels, 84 * in_channels)

        # 全连接层3
        # 这里的输入是84，输出是10
        self.fc3 = nn.Linear(84 * in_channels, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # x是一个4维的tensor，第一维是batch_size，第二维是通道数，第三维和第四维是图像的高和宽
        # 经过flatten操作后，x变成了2维的tensor，第一维是batch_size，第二维是通道数*图像的高*图像的宽
        # 因为下一层是全连接层（线性层），所以需要将图像展开成一维的
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义一个函数来显示图像
def imshow(img):
    # 输入的img是tensor类型，需要先转化为numpy类型
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()

    # plt.imshow()的输入是numpy类型
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def testTrainImgShow(train_loader_, classes_, batch_size_):
    # 随机获取训练图片
    dataiter_ = iter(train_loader_)
    images_, labels_ = dataiter_.next()

    # 显示图片
    # make_grid()将若干幅图像拼成一幅图像
    imshow(torchvision.utils.make_grid(images_))

    # 打印图片标签
    print(' '.join('%5s' % classes_[labels_[j]] for j in range(batch_size_)))


if __name__ == '__main__':
    # Step1 加载并标准化CIFAR10数据集

    # transforms.Compose()将多个transform组合起来使用
    # 这个transform的作用先是将PIL.Image或者numpy.ndarray转化为torch.FloatTensor
    # 然后将数值归一化到[-1,1]之间，即(x-mean)/std
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 设定batch_size, epochs
    batch_size = 128
    epochs = 90
    LR = 0.001

    # 先下载数据集进行计算std和mean
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    mean = train_set.data.mean(axis=(0, 1, 2)) / 255
    std = train_set.data.std(axis=(0, 1, 2)) / 255
    print("mean:", mean)
    print("std:", std)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    # train_loader是一个可迭代对象，每次迭代都会返回一个batch_size的数据
    # num_workers表示用几个子进程来加载数据
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # 下载并加载测试集
    # tag:测试集的归一化处理需要相同还是不同呢？
    test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 测试下训练集的图片
    # testTrainImgShow(train_loader, classes, batch_size)

    net = MyNet()

    # Step3 定义损失函数和优化器
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，使用随机梯度下降算法，学习率为0.001，动量为0.9
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    # 创建空列表来存储准确率和epoch
    accuracy_values = []
    epoch_numbers = []

    # Step4 训练网络
    for epoch in range(epochs):
        loop = tqdm.tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc=f'Epoch [{epoch + 1}/{epochs}]',
            ncols=100,
        )

        running_loss = 0.0
        # 获取训练数据
        for i, data in loop:
            inputs, labels = data
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # 反向传播，计算梯度，更新参数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loop.set_postfix(loss=running_loss / len(train_loader))

        # 每1个epoch记录一次准确率
        correct = 0
        total = 0
        # 因为加载的模型是训练好的，所以不需要计算梯度
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                # torch.max()返回两个tensor，第一个tensor是最大值，第二个tensor是最大值的索引
                # 这里我们只需要索引
                _, predicted = torch.max(outputs.data, 1)
                # labels.size(0)是一个batch中label的个数，也就是4
                total += labels.size(0)
                # (predicted == labels)返回一个tensor，tensor中的元素是True或者False
                # tensor.sum()将True转化为1，False转化为0，然后求和
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total  # 不需要使用 //，以保留小数
        formatted_accuracy = f'{accuracy:.2f}'  # 将准确率格式化为带两位小数的字符串
        print(f'Accuracy at epoch {epoch + 1}: {formatted_accuracy} %')
        accuracy_values.append(formatted_accuracy)  # 保存带两位小数的准确率
        epoch_numbers.append(epoch + 1)

    print('Finished Training')

    print(type[accuracy_values[0]])
    accuracy_values_float = [float(value) for value in accuracy_values]
    print(type[accuracy_values_float[0]])

    # 创建整体准确率随着epoch的变化图
    # 绘制曲线图，x轴是epoch，y轴是accuracy
    plt.plot(accuracy_values_float, label="Test Accuracy")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')

    # 先创建一个文件夹，用来存放结果
    # 如果没有这个文件夹，就创建一个
    save_dir = './results2/' + str(batch_size) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片
    plt.savefig(save_dir + 'accuracy_vs_epoch.png')

    # 保存那两个列表为csv文件
    np.savetxt(save_dir + 'accuracy_values.csv', accuracy_values_float, delimiter=',')
    np.savetxt(save_dir + 'epoch_numbers.csv', epoch_numbers, delimiter=',')

    plt.show()

    # 打印最佳准确率以及对应的epoch
    # 找到最大的准确率及其Index
    max_accuracy = max(accuracy_values_float)
    # 如果有多个最大值，返回所有的
    max_index = [index for index, value in enumerate(accuracy_values_float) if value == max_accuracy]
    print("max_accuracy:", max_accuracy)
    print("max_index:", max_index)



