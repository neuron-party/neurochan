import cv2
import gym
import time
import random
import pyautogui
import numpy as np
from mss import mss

from scoring import *


class Osu(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(100)
        
        # screen capture resolution (h, w)
        self.resolution = (600, 800, 3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.resolution
        )
        
        # parameters for screen capture
        self.bounding_box = bounding_box
        
        self.sc = mss()
        # current frame
        self.current_frame = np.array(self.sc.grab(self.bounding_box))[:, :, 0:3]
        self.last_score = 0
        self.done = False
        
    def step(self, action): # action 
        
        # self._apply_action(action)
        
        sct_img = self.sc.grab(self.bounding_box)
        frame = np.array(sct_img)[:, :, 0:3]
        
        # not the most optimal solution for a done screen?
        if sum(sum(sum(frame == done_screen))) >= 1400000:
            self.done = True
        
        # janky solution, but algorithm always recognizes the leading zeros as '3' when the screen dims upon 'map death' 
        # so i think we can take advantage of this for the environment to use as the done parameter
        score = frame[0:30, 675:800, :]
        score = int(get_score(score))
        
        if str(score)[0:4] == '3333':
            self.done = True
            return frame, self.last_score, self.done
            
        self.last_score = score
        return frame, score, self.done
    
    def reset(self):
        # 960, 535 - reset coordinate
        
        # need to click twice for some reason
        time.sleep(1)
        pyautogui.click(960, 535)
        time.sleep(1)
        pyautogui.click(960, 535)
        
        # time.sleep(5) ensures that screen capturing does not begin in the very early stages of the map,
        # where get_score() function struggles due to the dimness of the entire map as it is loading in
        time.sleep(5)
        
    
    def _apply_action(self, action):
        x, y = action # mouse click
        pyautogui.click(x, y)

    # def reset(self): 
    #     return