import gym
import numpy as np
from collections import deque


class McOsuWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frame_stack = deque(maxlen=k)
        
    def reset(self):
        state = self.env.reset()
        for i in range(self.k):
            self.frame_stack.append(state)
        return np.stack(self.frame_stack)
    
    def step(self, action):
        total_reward = 0
        done = False
        
        # k frame skips or end of episode
        for i in range(self.k):
            next_state, reward, d = self.env.step(action)
            self.frame_stack.append(next_state)
            total_reward += reward
            
            if d:
                done = True
                break
        return np.stack(self.frame_stack), total_reward, done