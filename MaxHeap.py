import math
import random
from collections import namedtuple

"""
    This is unused. I was just messing around with the rank-based prioritized
    experience replay. Decided to use SumTree instead. This implementation
    is also slow and not optimized since I was only using it for curiousity
    testing.
"""
Experience = namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'reward', 'error', 'done', 'new_state', 'priority'])
class MaxHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return math.floor((i - 1) / 2)

    def insert(self, experience):
        self.heap.append(experience)
        self._heapify(len(self.heap) - 1)

    def changePriority(self, index, priority):
        self.heap[index].priority = priority
        self._heapify(index)

    def _heapify(self, index):
        parentIdx = self.parent(index)
        curIdx = index
        while self.heap[parentIdx].priority < self.heap[curIdx].priority and curIdx != 0:
            # Swap parent and current index
            self.heap[parentIdx], self.heap[curIdx] = \
                self.heap[curIdx], self.heap[parentIdx]
            curIdx = parentIdx
            parentIdx = self.parent(parentIdx)

s = MaxHeap()
for i in range(5):
    s.insert(Experience('s', 's', 's', 0, 0, 's', random.randint(1, 100000)))

print(s.heap)