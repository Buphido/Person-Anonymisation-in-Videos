import math


def progress(desc, i, n, steps):
    if i > n or i < 1 or steps <= 0:
        return
    steps = math.floor(steps)
    if i > 1:
        percent = math.floor((i-1) / n * 100)
        for j in range(len(str(percent) + "% []") + steps):
            print("\b", end='')
    else:
        print(desc + " ", end='')
    percent = math.floor(i / n * 100)
    print(str(percent) + "% [", end='')
    for j in range(steps):
        if 100 / steps * j <= percent:
            print("■", end='')
        else:
            print("□", end='')
    print("]", end='')
    return