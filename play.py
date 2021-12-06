import gym
import torch as T
from standardDQN import DQN
from gym.wrappers import Monitor

#env = Monitor(gym.make('CartPole-v0'), './video', force=True)
env = Monitor(gym.make('IceHockey-ram-v0', render_mode='human'), './video', force=True)
net = DQN(0, env.action_space.n, env.observation_space.shape)
net.load_state_dict(T.load('icehockey_10M_DQN'))
net.eval()

#while True:
env.reset()
done = False
obs = T.tensor(env.reset(), dtype=T.float).to(net.device)
while not done:
    #env.render()
    #print(T.argmax(net.forward(obs)).item())
    #print(T.argmax(net.forward(obs)))
    obs, _, done, _ = env.step(T.argmax(net.forward(obs)).item())
    obs = T.tensor(obs, dtype=T.float).to(net.device)
env.close()