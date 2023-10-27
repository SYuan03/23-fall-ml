
# If you need to import additional packages or classes, please import here.

def func():
    num = int(input())
    node = []
    root = []
    for i in range(num):
        root.append(0)
    for i in range(num):
        node.append([])
        dependence = input().split(" ")
        de_num = int(dependence[0])
        for j in range(de_num):
            root[int(dependence[j + 1]) - 1] = 1
            node[i].append(int(dependence[j + 1]))
    r = []
    for i in range(num):
        if root[i] == 0:
            r.append(i + 1)
    if len(r) == 0:
        print(-1)
        return
    max_depth = -1

    for i in r:
        i_dep = depth(node, i)
        if i_dep > max_depth:
            max_depth = i_dep


    # print(node)
    # print(root)

    print(i_dep)

    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().



def depth(node, i):
    if len(node[i - 1]) == 0:
        return 1
    else:
        dep = 0
        for j in range(len(node[i - 1])):
            j_dep = depth(node, node[i - 1][j]) + 1
            if j_dep > dep:
                dep = j_dep
        return dep

if __name__ == "__main__":
    func()
