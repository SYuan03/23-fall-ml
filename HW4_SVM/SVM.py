from PicRead import read_idx3_ubyte_pixel_file
from PicRead import read_idx3_ubyte_label_file
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC, LinearSVC, NuSVC

train_data, (height, width) = read_idx3_ubyte_pixel_file("data/train-images-idx3-ubyte")
train_labels = read_idx3_ubyte_label_file("data/train-labels-idx1-ubyte")
test_data = read_idx3_ubyte_pixel_file("data/t10k-images-idx3-ubyte")[0]
test_labels = read_idx3_ubyte_label_file("data/t10k-labels-idx1-ubyte")

# print(train_data)
# print(train_labels)

train_features = []
test_features = []
for i in train_data:
    patches = hog(i, orientations=10)
    patches = patches.flatten()
    train_features.append(patches)
    # print(patches)
    # print(patches.shape)
for i in test_data:
    features = hog(i, orientations=10)
    features = features.flatten()
    test_features.append(features)
    # print(features)
    # print(features.shape)

model = LinearSVC()
# model.C = 0.9
# model.max_iter = 100
model.fit(train_features, train_labels)
result = model.predict(test_features)
cm = pd.DataFrame(
    confusion_matrix(test_labels, result),

)
print("Info : confusion matrix")
print(cm)
print("Info : precision score")
print(precision_score(test_labels, result, average=None))
print("Info : recall score")
print(recall_score(test_labels, result, average=None))
