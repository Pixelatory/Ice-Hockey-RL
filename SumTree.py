import numpy as np

import util

'''
    Data Structure where all nodes but leafs have
    a value which is the sum of children node
    values.
    
    Used http://www.sefidian.com/2021/09/09/sumtree-data-structure-for-prioritized-experience-replay-per-explained-with-python-code/
    as a reference as to how the structure works.
'''
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.index = 0

        self.tree = np.zeros(2 * capacity - 1)  # Container for the tree node containing priorities.
        self.data = np.zeros(capacity, dtype=util.Experience)  # Container for the experience of each tree node.

    # Propogate the difference upwards into the parents.
    def propagate(self, idx, difference):
        parent = (idx - 1) // 2

        self.tree[parent] += difference

        # Obviously, stop at the root node.
        if parent != 0:
            self.propagate(parent, difference)

    # Find the node with tree value closest to p.
    def find(self, idx, p):
        leftChild = 2 * idx + 1
        rightChild = leftChild + 1

        if leftChild >= len(self.tree):
            return idx

        if p > self.tree[rightChild]:
            return self.find(rightChild, p - self.tree[leftChild])
        else:
            return self.find(leftChild, p)

    # Total is just the priority of the root node.
    def total(self):
        return self.tree[0]

    # Store the priority and experience.
    def add(self, priority, experience):
        idx = self.index + self.capacity - 1

        self.data[self.index] = experience
        self.update(idx, priority)

        if self.count < self.capacity:
            self.count += 1

        self.index += 1
        if self.index >= self.capacity:
            self.index = 0

    # Update priority of node at idx and propagate upwards.
    def update(self, idx, priority):
        difference = priority - self.tree[idx]

        self.tree[idx] = priority
        self.propagate(idx, difference)

    # Return tuple of index, priority, and the experience object.
    def get(self, s):
        idx = self.find(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])