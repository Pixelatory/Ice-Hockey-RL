from collections import deque, namedtuple
from random import sample

import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from util import plot_learning_curve, Experience

# from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)
# end of from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c


class DQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        return self.model(state)


class Agent:
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0,
                 eps_iter=1000000, eps_min=0.1, experience_size=1000000):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_iter = eps_iter
        self.action_space = [i for i in range(self.n_actions)]
        self.experiences = ExperienceReplay(experience_size)

        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        print(self.device)
        self.Q = DQN(self.lr, self.n_actions, self.input_dims).to(self.device)
        self.targetQ = DQN(self.lr, self.n_actions, self.input_dims).to(self.device)
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.targetQ.eval()

        self.learnCounter = 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.device)
            qValues = self.Q.forward(observation)
            action = np.argmax(qValues.to('cpu').detach().numpy())
        else:
            action = np.random.choice(self.action_space)

        self.decrement_epsilon()

        return action

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - (1 / self.eps_iter)

    def learn(self, batch, targetSetIter):
        batch = Experience(*zip(*batch))

        states = T.tensor(batch.state, dtype=T.float).to(self.device)
        actions = T.tensor(batch.action, dtype=T.int).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float).to(self.device)
        states_ = T.tensor(batch.new_state, dtype=T.float).to(self.device)
        dones = T.tensor(batch.done, dtype=T.int).to(self.device)

        curr_Q = self.Q.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        target_Q = self.targetQ.forward(states_)
        max_target_Q = T.max(target_Q, 1)[0]
        # Notice: dones are the int of done
        target_Q = rewards.squeeze(1) + (self.gamma * max_target_Q * dones)

        self.Q.zero_grad()
        loss = self.Q.loss(curr_Q, target_Q)
        loss.backward()
        self.Q.optimizer.step()

        # Reset the target network weights to online network
        if self.learnCounter % targetSetIter:
            self.targetQ.load_state_dict(self.Q.state_dict())
            self.targetQ.eval()
            self.learnCounter = 1
            return

        self.learnCounter += 1


if __name__ == '__main__':
    #env = gym.make('IceHockey-ram-v0', render_mode='rgb_array')
    env = gym.make('CartPole-v0')

    # HYPERPARAMETERS
    experience_size = 1000000
    targetNetworkIter = 1000  # How many iters to set neural network weights
    experience_sample_size = 32  # How many experiences to sample
    max_explore_iter = 50000  # Maximum amount of iterations to explore
    max_frames = 100000  # Total frames of play time
    replay_start_size = 10000  # Select random actions until experience has this many elements in it
    learning_rate = 5e-3

    print("EXPERIENCES_SIZE:", experience_size)
    print("TARGET_NETWORK_UPDATE:", targetNetworkIter)
    print("MINIBATCH SIZE:", experience_sample_size)
    print("FINAL EXPLORATION FRAME:", max_explore_iter)
    print("MAX FRAMES:", max_frames)
    print("REPLAY START SIZE:", replay_start_size)
    print("LEARNING RATE:", learning_rate)
    print("INPUT_DIM:", env.action_space.n)
    print("OUTPUT_DIM:", env.observation_space.shape[0])

    # Used for outputting data.
    scores = []
    eps_history = []
    frame = 0
    episode = 0

    agent = Agent(lr=learning_rate,
                  input_dims=env.observation_space.shape[0],
                  n_actions=env.action_space.n,
                  eps_iter=max_explore_iter,
                  experience_size=experience_size)

    # tqdm used for estimating completion time, and seeing speed of algorithm.
    pbar = tqdm(total=max_frames)
    while frame < max_frames:
        score = 0
        done = False
        obs = env.reset()
        while not done:
            pbar.update(1)
            frame += 1

            if len(agent.experiences) < replay_start_size:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward

            # [reward] for rewards.squeeze(1) in agent.learn()
            # not done for calculating target Q value in agent.learn()
            agent.experiences.append(Experience(obs, action, [reward], not done, obs_))

            if len(agent.experiences) > replay_start_size:
                experience_sample = agent.experiences.sample(experience_sample_size)
                agent.learn(experience_sample, targetNetworkIter)

            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        if episode % 5 == 0:
            avg_score = np.mean(scores[-5:])
            print('Episode ' + str(episode) + ": ", 'score %.1f average score %.1f epsilon %.2f' %
                  (score, avg_score, agent.epsilon))
        episode += 1
    pbar.close()

    T.save(agent.Q.state_dict(), 'cartpole_model')
    filename = 'cartpole_model.png'
    x = [i + 1 for i in range(episode)]
    plot_learning_curve(x, scores, eps_history, filename)
