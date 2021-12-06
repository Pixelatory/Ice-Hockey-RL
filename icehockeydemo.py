import gym
import numpy
from gym.wrappers import Monitor

env = gym.make('IceHockey-ram-v0', render_mode='human')
#env = Monitor(gym.make('IceHockey-v0', render_mode='human'), './video', force=True)
env.reset()
t = 0

#action = print(env.action_space.sample())
while True:
    t += 1
    #action = env.action_space.sample()
    state, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    if done:
        print("Episode finished after {} time steps".format(t + 1))
        break
"""
0 -> nothing
1 -> stationary_swipe
2 -> up
3 -> right
4 -> left
5 -> down
6 -> up_right
7 -> up_left
8 -> down_right
9 -> down_left
10 -> swipe
11 -> right_swipe
12 -> left_swipe
13 -> down_swipe
14 -> up_right_swipe
15 -> up_left_swipe
16 -> down_right_swipe
17 -> down_left_swipe

reward signal:
1 -> if we score
-1 -> if opponent scores
"""
env.close()
