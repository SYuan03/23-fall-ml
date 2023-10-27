import copy
import csv

dataPath = "./dataResource/"


def train_file_read(path):
    re = []
    with open(path, newline='', encoding="UTF-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        # 训练
        classify_num = 3
        useful_count = []
        year = []
        result = []
        # 修正出现次数过少的Type 易产生误差
        for row in reader:
            row.remove(row[0])
            row.remove(row[0])
            row.remove(row[1])
            # 取年份作为分的类
            print(row)
            row[1] = int(row[1].split(" ")[2])
            row[2] = int(row[2])
            result.append(int(row[-1]))
            row.remove(row[-1])
            print(row)
            # 将
            useful_count.append(int(row[2]))
            year.append(int(row[1]))

            re.append(row)
        type_enum = []
        type_enum_num = []
        for row in re:
            if not type_enum.__contains__(row[0]):
                type_enum.append(row[0])
                type_enum_num.append(0)
            type_enum_num[type_enum.index(row[0])] += 1
            row[1] = str(row[1])
            # 对count进行区间化
            if row[2] < 10:
                row[2] = "x < 5"
            elif row[2] < 20:
                row[2] = "5 <= x < 20"
            elif row[2] < 40:
                row[2] = "20 <= x <= 50"
            elif row[2] < 50:
                row[2] = "40 <= x < 60"
            else:
                row[2] = "x >= 60"
        # 将药品类别进行预剪枝，减少样本数量过少导致的过拟合
        temp = copy.deepcopy(type_enum)
        for i in range(len(type_enum)):
            if type_enum_num[i] < 10:
                type_enum.remove(temp[i])
        for row in re:
            if not type_enum.__contains__(row[0]):
                row[0] = "Other"
        return re, result

def test_file_read(path):
    re = []
    with open(path, newline='', encoding="UTF-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        # 训练
        classify_num = 3
        useful_count = []
        year = []
        result = []
        # 修正出现次数过少的Type 易产生误差
        for row in reader:
            row.remove(row[0])
            row.remove(row[0])
            row.remove(row[1])
            # 取年份作为分的类
            print(row)
            row[1] = int(row[1].split("-")[2])
            row[2] = int(row[2])
            # result.append(int(row[-1]))
            row.pop()
            print(row)
            # 将
            useful_count.append(int(row[2]))
            year.append(int(row[1]))

            re.append(row)
        type_enum = []
        type_enum_num = []
        for row in re:
            if not type_enum.__contains__(row[0]):
                type_enum.append(row[0])
                type_enum_num.append(0)
            type_enum_num[type_enum.index(row[0])] += 1
            row[1] = str(row[1])
            # 对count进行区间化
            if row[2] < 10:
                row[2] = "x < 5"
            elif row[2] < 20:
                row[2] = "5 <= x < 20"
            elif row[2] < 40:
                row[2] = "20 <= x < 40"
            elif row[2] < 50:
                row[2] = "40 <= x < 50"
            else:
                row[2] = "x >= 50"
        # 将药品类别进行预剪枝，减少样本数量过少导致的过拟合
        temp = copy.deepcopy(type_enum)
        for i in range(len(type_enum)):
            if type_enum_num[i] < 10:
                type_enum.remove(temp[i])
        for row in re:
            if not type_enum.__contains__(row[0]):
                row[0] = "Other"
        return re


def write_test_result(path, result):
    re = []
    with open(path, newline='', encoding="UTF-8") as f:
        with open("./dataResource/result.csv", "w", newline="") as target_file:
            reader = csv.reader(f)
            header = next(reader)
            writer = csv.writer(target_file, delimiter=',')
            writer.writerow(header)
            i = 0
            for row in reader:
                row.pop()
                row.append(result[i])
                i += 1
                writer.writerow(row)


def calculate_EntD(data, result, n):
    # H(x) = sum(-p(x)log_2(p(x))) for x
    # 给定数据集中，x共有五种取值
    pass

def find_classify_node(data, result, num):
    # 分为num个类，就是找 num - 1个划分点
    if len(data) != len(result):
        print("Error : data length not equals result length")
        return
    else:
        length = len(data)
    classify_pod = []
    sorted_data = list(data).sort()
    # 采用局部最优解的情况，依次找出 num - 1个划分点
    # 计算熵增

    for i in range(num - 1):
        index = -1
        entropy = -1
        for j in range(len(result)):

            pass
    pass
if __name__ == '__main__':
    train_file_read()
    pass
