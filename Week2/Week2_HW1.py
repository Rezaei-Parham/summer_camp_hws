
import numpy as np
import gym
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from torch.nn.functional import one_hot


device = torch.device("cuda")

env = gym.make("FrozenLake-v1", is_slippery=False, new_step_api=True)

Transition = namedtuple('transition',('state','action','next_state','reward','done'))




class Mem:
  def __init__(self, size):
    self.mem_size = size
    self.memory = deque([],maxlen=size)

  def push(self, *args):
    self.memory.append(Transition(*args))

  def sample(self,batch_size):
    return random.sample(self.memory,batch_size)

  def __len__(self):
    return len(self.memory)


class DQN(nn.Module):
  def __init__(self,in_size,out_size):
    super(DQN, self).__init__()
    #duelingDQN
    self.feauture_layer = nn.Sequential(
        nn.Linear(in_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU()
    )

    self.value_stream = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    self.advantage_stream = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, out_size)
    )

  def forward(self, state):
      features = self.feauture_layer(state)
      values = self.value_stream(features)
      advantages = self.advantage_stream(features)
      qvals = values + (advantages - advantages.mean())

      return qvals

total_episodes = 20000
max_steps = 500
learning_rate = 1e-3
gamma = 1
batch_size = 256
target_update = 10
train_frequency = 1
train_epochs = 10
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0001
TAU = 0.0005

class RLGame():
  def __init__(self,env,epsilon):
    self.nspace=env.observation_space.n
    self.naction=env.action_space.n
    self.env = env
    self.policy_net = DQN(self.nspace,self.naction).to(device)
    self.target_net = DQN(self.nspace,self.naction).to(device)
    self.reload_target()
    self.target_net.eval()
    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate,amsgrad=True)
    self.memory = Mem(10000)
    self.one_hot_encoder = one_hot(torch.arange(0,self.nspace)) * 1.
    self.loss = nn.SmoothL1Loss()
    self.epsilon = epsilon

  def reload_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def get_action(self,state,eval):
      if random.uniform(0,1) > self.epsilon or eval:
        with torch.no_grad():
          u1 = self.policy_net(state)
          action = u1.max(1)[1].view(1,1)
          return action

      return torch.tensor([self.env.action_space.sample()],device=device).view(1,1)

  def ohe(self,x):
    return self.one_hot_encoder[x]

  def take_steps(self):
    state = self.env.reset()
    state = self.ohe(state).to(device).view(1,16)
    total_rewards = 0
    for _ in range(max_steps):

        action = self.get_action(state,False)
        next_state, reward, terminated, truncated, _ = self.env.step(action.item())

        done = terminated or truncated

        if reward > 0:
          reward = 1000
          print(state,next_state,done,reward)

        if done:
          if reward <= 0:
            reward = -5

        done = torch.tensor([int(done)], device=device)
        next_state = self.ohe(next_state).to(device).view(1,16)


        if (not done) and torch.equal(next_state,state):
          reward = -2
        total_rewards += reward
        repeat = False
        if reward > 0:
          repeat = True
        reward = torch.tensor([reward], device=device)
        state = next_state

        self.memory.push(state, action, next_state, reward, done)
        if repeat:
          for i in range(10):
            self.memory.push(state, action, next_state, reward, done)
        if terminated:
            break
    return total_rewards


  def train(self):
    rewards = []
    for episode in range(1, total_episodes+1):


        self.policy_net.eval()
        total_rewards = self.take_steps()

        self.policy_net.train()
        if episode % train_frequency == 0 and len(self.memory) >= batch_size:
          for _ in range(train_epochs):
            transitions = self.memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.cat(batch.done)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            with torch.no_grad():
              next_state_index = self.target_net(next_state_batch).max(1)[1].detach()

            target_q_values = (self.policy_net(next_state_batch)[:,next_state_index]  * gamma * (1-done_batch)) + reward_batch
            #target_q_values = (next_state_index  * gamma * (1-done_batch)) + reward_batch

            loss = self.loss(state_action_values.squeeze(1), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



          target_net_state_dict = self.target_net.state_dict()
          policy_net_state_dict = self.policy_net.state_dict()
          for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
          self.target_net.load_state_dict(target_net_state_dict)

        self.epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(total_rewards)


        print(f"Episode {episode}: Reward = {total_rewards}, Epsilon = {self.epsilon}")

r = RLGame(env,epsilon)
r.train()

r.get_action(r.ohe(1).to(device).unsqueeze(0),True).item()

from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob
import base64, io, os

os.environ['SDL_VIDEODRIVER']='dummy'
os.makedirs("video", exist_ok=True)

def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

def show_video_of_model(env_name):
    vid = video_recorder.VideoRecorder(env, path="video/{}.mp4".format(env_name))
    done = False
    state=2
    r.policy_net.eval()
    for t in range(max_steps):
        vid.capture_frame()
        with torch.no_grad():
          action = r.get_action(r.ohe(state).to(device).unsqueeze(0),True).item()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"state: {state}, action: {action}",next_state, reward, done)
        state = next_state
        if done:
            break
    vid.close()
    env.close()


r.policy_net.eval()
for i in range(16):
  print(r.policy_net.forward(r.ohe(i).to(device)))



env.reset()
env.step(2)
env.step(2)


show_video_of_model("FrozenLake-v1")



show_video("FrozenLake-v1")


import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
from tqdm import trange
os.environ["SDL_VIDEODRIVER"] = "dummy"
clear_output()


Actions =  {0: 'UP',1: 'RIGHT',2: 'DOWN',3: 'LEFT'}



def visualize(env, action=None, reward=None):
    env_screen = env.render(mode = 'rgb_array')
    plt.imshow(env_screen)
    plt.axis('off');
    title = ''
    if action:
        title += f'Action: {Actions[action]}'
    if reward:
        title += f'Reward: {reward}'

    plt.title(title)
    plt.show()



class Agent:

    def __init__(self, env, noise):
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = {}
        self.env = env
        self.noise = noise
        self.current_state = None

    def learn(self, num_episodes, alpha, gamma, epsilon):
        raise NotImplementedError()

    def create_policy(self):
        for state in range(len(self.q_values)):
            self.policy[state] = np.argmax(self.q_values[state])

class CliffWalkerQL(Agent):

    def learn(self, num_episodes, alpha, gamma, epsilon):

        for episode in trange(num_episodes):
            state = env.reset()
            while True:
                action = np.argmax(self.q_values[state])

                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.env.action_space.n - 1)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.q_values[state, action] += alpha * (reward + gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])

                state = next_state

                if done:
                    break


# In[410]:


cliff_walker_ql = CliffWalkerQL(env, 0)
alpha = 0.8
gamma = 0.95
epsilon = 0.2
episodes = 10000
cliff_walker_ql.learn(episodes, alpha, gamma, epsilon)


from time import sleep

cliff_walker_ql.create_policy()
state = env.reset()
epochs, penalties, reward = 0, 0, 0

done = False

while not done:
    action = cliff_walker_ql.policy[state]
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if reward == -10:
        penalties += 1

    epochs += 1
    visualize(env)

    print(f"Timestep: {epochs}")
    print(f"State: {state}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    sleep(1)





