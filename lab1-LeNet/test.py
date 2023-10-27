import torchvision

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)

print(train_set.data.shape)
print(train_set.data)

# 计算std和mean
# axis=(0, 1, 2)表示对每个通道的所有像素点求std和mean
mean = train_set.data.mean(axis=(0, 1, 2)) / 255