import math

def calculate_entropy(result, re_enum):
    re = []
    entropy = 0
    for i in range(len(re_enum)):
        re.append(0)
    for i in range(len(result)):
        re[list(re_enum).index(result[i])] += 1
    for i in range(len(re_enum)):
        pi = re[i] / sum(re)
        entropy -= pi * math.log2(pi)
    return entropy

def calculate_entropy_gain(data, result):
    pass

def calculate_gain_sa(data, enum, result, re_enum):
    # print(data)

    gain = 0
    re = []
    for i in range(len(enum)):
        re.append([])
        for j in range(len(re_enum)):
            re[i].append(0)

    for i in range(len(data)):
        # print(list(enum).index(data[i]))
        # print(list(re_enum).index(result[i]))
        # print(re)
        re[list(enum).index(data[i])][list(re_enum).index(result[i])] += 1
        # 获得频率矩阵
    s_size = len(data)
    for i in range(len(enum)):
        sv_size = sum(re[i])
        entropy = 0
        for j in range(len(re_enum)):
            pi = re[i][j] / sv_size
            if pi == 0.0:
                entropy += 999
            else:
                entropy -= pi * math.log2(pi)
        gain -= entropy * sv_size / s_size
    return gain

    pass