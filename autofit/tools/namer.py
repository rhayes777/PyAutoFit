"""
Singleton to provide globally unique names

namer("hello") -> "hello0"
namer("hello") -> "hello1"
namer("world") -> "hello"
"""

from collections import Counter


class Namer:
    def __init__(self):
        self.counter = Counter()

    def __call__(self, name):
        number = self.counter[name]
        self.counter[name] += 1
        return f"{name}{number}"

    def reset(self):
        self.counter.clear()


namer = Namer()
