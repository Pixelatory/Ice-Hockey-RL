import random
import numpy as np

from SumTree import SumTree

'''
    In the SumTree, store Experience (from util),
    along with the priority. SumTree has intermediate
    node values as the sum of children, and so
    the cumulative priority is easily determined.
    
    This is essentially another Experiences container
    like in standard DQN, but now it's the prioritized
    memory that uses a SumTree.
'''
class PrioritizedMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

        # Chose alpha and beta from the paper
        self.alpha = 0.6
        self.beta = 0.4
        self.eps = 0.001

    def add(self, error, sample):
        p = self.priority(error)
        self.tree.add(p, sample)

    def update(self, idx, error):
        p = self.priority(error)
        self.tree.update(idx, p)

    def priority(self, error):
        return (error + self.eps) ** self.alpha

    # Return a sample, but segment the data by total cumulative priority first.
    def sample(self, n):
        batch = []
        indexes = []
        segmentRatio = self.tree.total() / n
        priorities = np.zeros(n)

        for i in range(n):
            lProbBound = segmentRatio * i
            hProbBound = segmentRatio * (i + 1)

            (idx, priority, data) = self.tree.get(random.uniform(lProbBound, hProbBound))
            priorities[i] = priority
            batch.append(data)
            indexes.append(idx)

        probabilities = priorities / self.tree.total()
        importanceWeight = np.power(probabilities * self.tree.count, -self.beta)
        importanceWeight /= importanceWeight.max()  # Divide each entry by maximal weight
        return batch, indexes, importanceWeight.tolist()

    def __len__(self):
        return self.tree.count
