def f(x):

    y = max(y)
    def zero(t):
        if t(x) > 0:
            return y
        return 0
    return zero

y = 1
while y < 10:
    if f(y)(lambda z : z - y + 10):
        max = y
    y = y + 1
print(max)

