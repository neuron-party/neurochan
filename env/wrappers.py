import cv2
import gym
import numpy as np
from collections import deque


class McOsuWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frame_stack = deque(maxlen=k)
        
    def reset(self):
        frame = self.env.reset()
        state = self._process(frame)
        
        for i in range(self.k):
            self.frame_stack.append(state)
            
        return np.stack(self.frame_stack)
    
    def step(self, action):
        total_reward = 0
        done = False
        
        # k frame skips or end of episode
        for i in range(self.k):
            next_frame, reward, d, _ = self.env.step(action)
            next_state = self._process(next_frame)
            self.frame_stack.append(next_state)
            total_reward += reward
            
            if d:
                done = True
                break
                
        return np.stack(self.frame_stack), total_reward, done
    
    def _process(self, frame):
        '''
        downscale and normalize frame for memory/computation efficiency
        '''
        state = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dsize=(80, 80))
        return state