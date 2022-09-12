import random
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from utils.replay import *
# from networks.flexnet import * 
# from networks.deepmind import *

'''
Implementation of a Double Deep Q Learning Agent (DDQN)
For environments with small, discrete action spaces, Q-learning approaches are able to effectively converge to optimality
Q(s, a) = Q(s, a) + R(s, a, s') * γ * max(Q(s', a'))
    Q(s, a) : values given the current state and selected actions
    max(Q(s', a')) : values of optimal actions given the next state
    
    Q_predicted : Q(s, a)
    Q_target : max(Q(s', a'))
    
The agent converges to optimality by minimizing the loss given by the temporal difference:
    temporal difference = R(s, a, s') * γ * max(Q(s', a')) - Q(s, a)
    temporal difference = Q_target - Q_predicted
    loss = SmoothL1Loss(Q_predicted, Q_target)
'''
class DQNAgent:
    def __init__(self, observation_space, action_space, **params):
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # agent parameters
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        
        # by-frame epsilon calculation
        self.eps_start = params['eps_start']
        self.eps_interval = params['eps_interval'] 
        self.eps_ff = params['eps_ff'] # epsilon final frame
        
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.memory_size = params['memory_size']
        self.num_model_updates = 0
        self.target_net_updates = params['target_net_updates']
        
        # network parameters
        self.network_params = params['network_params']
        self.batch_size = params['batch_size']
        self.device = torch.device(params['device'])
        self.dtype = torch.float32
        self.updates = 0
        
        self._build_agent()

    def _build_agent(self):
        self.replay = ExperienceReplay(self.memory_size)
        #self.replay = GPUExperienceReplay(self.memory_size, device='cuda:0')
        
        '''Preconfigured Deepmind CNN Architecture'''
        self.network = DeepmindCNN().to(self.device)
        self.target_network = DeepmindCNN().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        '''Flexible user-specified network'''
        # self.network = Net(*self.network_params).to(self.device)
        # self.target_network = Net(*self.network_params).to(self.device)
        # self.target_network.load_state_dict(self.network.state_dict())
        self.optim = optim.Adam(self.network.parameters(), lr=self.alpha)
        self.loss = torch.nn.SmoothL1Loss()
    
    def action(self, state):
        if self.epsilon < random.random():
            actions = self.network(torch.tensor(state, device=self.device, dtype=self.dtype).unsqueeze(0) / 255)
            actions = actions.detach().cpu().numpy()
            action = np.argmax(actions)
        else:
            if np.random.rand() > 0.5:
                action = np.random.randint(0, 30000)
            else:
                action = 30000
        return action
        
    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
    
    # by-frame epsilon updates
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.eps_start - (self.eps_interval * self.num_model_updates / self.eps_ff))
                                      
    def update(self, update=1):
        for e in range(update):
            self.optim.zero_grad()
            minibatch = self.replay.get(self.batch_size)
            
            # storing frames on the RAM
            if (isinstance(self.replay, ExperienceReplay)):
                # minibatch should be numpy arrays
                obs = torch.stack([i[0] for i in minibatch]).to(self.device).to(self.dtype) / 255
                next_obs = torch.stack([i[3] for i in minibatch]).to(self.device).to(self.dtype) / 255
                actions = np.stack([i[1] for i in minibatch]).reshape(-1, 1)
                actions = torch.tensor(actions).to(self.device).to(torch.int64)
                rewards = torch.tensor([i[2] for i in minibatch]).to(self.device)
                dones = torch.tensor([i[4] for i in minibatch]).to(self.device)

            # storing frames on the GPU
            elif (isinstance(self.replay, GPUExperienceReplay)):
                # minibatch should be holding gpu tensors
                obs = torch.stack([i[0] for i in minibatch]) / 255
                actions = torch.stack([i[1] for i in minibatch])    
                rewards = [i[2] for i in minibatch]
                next_obs = torch.stack([i[3] for i in minibatch]) / 255
                dones = [i[4] for i in minibatch]
            
            Qs = self.network(torch.cat([obs, next_obs]))
            Q_s0, Q_s1 = torch.split(Qs, self.batch_size, dim=0)
            
            # action is always chosen by original Q
            a1 = torch.argmax(Q_s1, dim=1).reshape(-1, 1)
            Q_s1_p = self.target_network(next_obs)
            
            Q_predicted = torch.gather(Q_s0, 1, actions).squeeze() # values of chosen actions
            Q_prime = torch.gather(Q_s1_p, 1, a1).squeeze() # values of optimal actions 
            
            Q_actual = rewards + self.gamma * Q_prime * dones # temporal difference error 
            
            # loss
            loss = self.loss(Q_predicted, Q_actual) 
            loss.backward()
            self.optim.step()

            self.num_model_updates += 1
            
            # update target Q network to current Q network
            if self.num_model_updates % self.target_net_updates == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                self.updates += 1

        # traditional epsilon decay
        if self.epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
    def save(self, path):
        torch.save({'network': self.network.state_dict(),
                    'optim': self.optim.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
        
        if self.target_net_updates is not None:
            self.target_network = copy.deepcopy(self.network)
        
        self.epsilon = .01 # for testing
        
        
import numpy as np
import torch

# frames stored on RAM
class ExperienceReplay:
    def __init__(self, size):
        self.memory = []
        self.size = size
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        # transforms into 1s and 0s : 0 -> terminal state
        done = int(not done) 
        state, next_state = torch.tensor(state), torch.tensor(next_state)
        data = (state, action, reward, next_state, done)
        if (self.position >= len(self.memory)):
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.size
    
    def get(self, batch_size):
        if len(self.memory) - 1 == 0:
            indices = np.zeros(batch_size).astype(int)
        else:
            indices = np.random.randint(0, len(self.memory) - 1, batch_size)
        batch = [self.memory[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.memory)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# predefined networks
class DeepmindCNN(nn.Module):
    def __init__(self):
        super().__init__()

        network = [
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 30001)
        ]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x): 
        actions = self.network(x)
        return actions