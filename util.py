from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

Experience = namedtuple('Experience',
                        field_names=['state', 'action',
                                     'reward', 'done', 'new_state'])

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, label="1")
    ax2 = fig.add_subplot(1, 1, 1, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Epsilon", color="C0")

    N = len(scores)
    running_average = np.empty(N)
    for i in range(N):
        running_average[i] = np.mean(scores[0:(i + 1)])

    ax2.scatter(x, running_average, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color="C1")

    plt.savefig(filename)