import queue


# If you need to import additional packages or classes, please import here.

def func():

    resources = []
    size_left, size_right = map(int, input().split(" "))
    commands = int(input())
    for i in range(size_left, size_right + 1):
        resources.append(i)
    # print(resources)
    # 初始化
    for i in range(commands):
        command = input().split(" ")
        if command[0] == "3":
            id = int(command[1])
            if (not resources.__contains__(id)) and size_left <= id <= size_right:
                resources.append(id)
        if command[0] == "1":
            num = int(command[1])
            if len(resources) >= num:
                resources = resources[num:]
            continue
            # 从头开始调用
        if command[0] == "2":
            id = int(command[1])
            if resources.__contains__(id) and size_left <= id <= size_right:
                resources.remove(id)
            continue
    print(resources[0])
    pass
    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().

if __name__ == "__main__":
    func()
