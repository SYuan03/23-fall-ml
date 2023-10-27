
# If you need to import additional packages or classes, please import here.

def func():
    require_type = int(input())
    types = int(input())
    location = []
    for i in range(types):
        x, y = map(int, input().split(" "))
        location.append([x, y])
    max_dis = []
    for i in range(types):
        distance = []
        for j in range(types):
            if j != i:
                distance.append(max(abs(location[i][0] - location[j][0]), abs(location[i][1] - location[j][1])))
        distance.sort()
        # print(distance)
        # print(len(distance))
        max_dis.append(distance[require_type - 2])
    index = max_dis.index(min(max_dis))
    max_distance = max_dis[index]
    print(max_distance // 2)

    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().

if __name__ == "__main__":
    func()
