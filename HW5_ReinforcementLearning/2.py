
def jiech(n):
    result = 1
    for i in range(n):
        result = result * n
    return result

def ca(m, n):
    print(jiech(n) * jiech(n - m) / jiech(m))
    return (jiech(n) * jiech(n - m)) / jiech(m)

n, m = map(int, input().split(" "))
sum = 0
for i in range(1, m):
    sum += ca(i, n) * (ca(i, m) * (n - i) ** i)
print(sum)
