from math import floor


class Progress:
    def __init__(self, desc, n, steps):
        if desc is not None:
            self.desc = desc + ' '
        else:
            self.desc = ''
        self.i = 0
        if n is not None:
            self.n = n
        else:
            self.n = 0
        if steps is not None:
            self.steps = steps
        else:
            self.steps = 100
        self.init = False

    def initialise(self, n):
        if self.init:
            return
        if n is not None:
            self.n = n
        self.init = True
        print(self.desc + '0% [', end='')
        for step in range(self.steps):
            print('□', end='')
        print(']', end='')

    def update(self):
        if self.i > self.n or self.i < 0 :
            return
        if not self.init:
            self.initialise(None)
        percent = floor(self.i / self.n * 100)
        for j in range(len(str(percent) + '% []') + self.steps):
            print('\b', end='')
        self.i += 1
        percent = floor(self.i / self.n * 100)
        print(str(percent) + '% [', end='')
        for step in range(self.steps):
            if 100 / self.steps * step <= percent:
                print('■', end='')
            else:
                print('□', end='')
        print(']', end='')
        return

    def quit(self):
        if self.i != -1:
            self.i = -1
            print()
        return
