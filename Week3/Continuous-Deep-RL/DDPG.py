
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make('MountainCarContinuous-v0')


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_size, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_size)
        self.l1.weight.data.normal_(0, 1e-1)
        self.l2.weight.data.normal_(0, 1e-1)
        self.l3.weight.data.normal_(0, 1e-2)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_size + action_size, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)
        self.l1.weight.data.normal_(0, 1e-1)
        self.l2.weight.data.normal_(0, 1e-1)
        self.l3.weight.data.normal_(0, 1e-2)


    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# In[72]:


Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

class Memory(object):

    def __init__(self, capacity,batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size=batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
      transitions = random.sample(self.memory, self.batch_size)
      return  Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

MEM_SIZE = int(1e6)
BS = 64
GAMMA = 0.99
TAU = 1e-3
EPISODES = 100

class DDPG_TRAINER:
    def __init__(self, state_size, action_size, tau):
        self.memory = Memory(MEM_SIZE, BS)
        self.tau = tau
        self.std = 1
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def update_std(self):
        self.std = max(self.std - 0.01, 0.1)

    def get_action(self, state, add_N = True):
        state =  torch.tensor(state).to(device).float()
        action = self.actor(state).cpu().data.numpy()
        if add_N:
            action = action + np.random.normal(0, self.std)

        action[0] = np.clip(action[0],-1,1)
        return action

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.memory.batch_size:
            state, action, reward, next_state, done = self.memory.sample()

            state = torch.tensor(state).to(device).float()
            next_state = torch.tensor(next_state).to(device).float()
            reward = torch.tensor(reward).to(device).float()
            action = torch.tensor(action).to(device)
            done = torch.tensor(done).to(device).int()

            #critic
            target_action = self.actor_target(next_state)

            y = self.critic_target(next_state, target_action).detach()
            target_q = reward.unsqueeze(1) + (GAMMA*y*((1-done).unsqueeze(1)))
            critic_loss = F.mse_loss(self.critic(state, action), target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            #actor

            action_mu = self.actor(state)
            actor_loss = -self.critic(state, action_mu).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)

action_size = env.action_space.shape[0]
print(f'size of eche action = {action_size}')
state_size = env.observation_space.shape[0]
print(f'size of state = {state_size}')

trainer = DDPG_TRAINER(state_size = state_size, action_size = action_size, tau = TAU)

reward_list = []
for i in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = trainer.get_action(state)
        next_state, reward, done, _ = env.step(action)
        trainer.update(state, action, reward, next_state, done)
        total_reward +=reward
        state = next_state

    reward_list.append(total_reward)
    trainer.update_std()

    print(f"episode: {i+1}, current reward: {total_reward}")


# NEW METHOD DDPG

# In[80]:


# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob
import base64, io, os

os.environ['SDL_VIDEODRIVER']='dummy'


# In[85]:


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
    state = env.reset()
    done = False
    score = 0
    for t in range(500):
        vid.capture_frame()
        action= trainer.get_action(state,False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        score = reward
        if done:
            break
    print(score)
    vid.close()
    env.close()

show_video_of_model('MountainCarContinuous-v0')
show_video('MountainCarContinuous-v0')

