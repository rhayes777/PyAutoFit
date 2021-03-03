from collections import Counter


class Namer:
    def __init__(self):
        self.counter = Counter()

    def __call__(self, name):
        number = self.counter[name]
        self.counter[name] += 1
        return f"{name}{number}"


namer = Namer()