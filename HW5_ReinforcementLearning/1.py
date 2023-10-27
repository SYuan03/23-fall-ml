x, y, n = map(int, input().split(" "))
li = []
step = []

for i in range(n):
    xn, yn, sn = map(int, input().split(" "))
    li.append([xn, yn])
    step.append(sn)

move = [[0, -1], [0, 1], [1, 0], [-1, 0]]
current = [x - 1, y - 1]
possible = [[0, 0]]
st_need = []
if li.__contains__(possible[0]):
    st_need.append(step[li.index(possible[0])])
else:
    st_need.append(0)
    li.append([0, 0])
    step.append(1)
for t in range(100):
    for i in possible:
        for j in move:
            # 这个点可通行
            next = [i[0] + j[0], i[1] + j[1]]
            if li.__contains__(next):
                if possible.__contains__(next):
                    if st_need[possible.index(next)] > st_need[possible.index(i)] + step[li.index(i)]:
                        st_need[possible.index(next)] = st_need[possible.index(i)] + step[li.index(i)]
                else:
                    possible.append(next)
                    st_need.append(st_need[possible.index(i)] + step[li.index(i)])

if possible.__contains__(current):
    if current == [0, 0]:
        print(0)
    else:
        print(st_need[possible.index(current)])
else:
    print(-1)