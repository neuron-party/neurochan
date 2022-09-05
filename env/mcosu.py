'''gym wrapper for McOsu, a remake of osu! but without anticheats preventing pyautogui/win32api from moving the cursor around'''
import cv2
import gym
import time
import random
import pyautogui
import win32api
import pydirectinput
import numpy as np
from mss import mss
from collections import deque

from utils.scoring import *
from utils.click import *


class McOsu(gym.Env):
    def __init__(self, bounding_box, hardcode_list, done_screen):
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
        self.hardcode_list = hardcode_list
        self.done_screen = done_screen
        
        # apply a margin to the screen since no circles appear on the very left/right/top/bottom of the map
        margin = 30
        self.xl_bound = self.bounding_box['left'] + margin
        self.xr_bound = self.bounding_box['left'] + self.bounding_box['width'] - margin
        self.yt_bound = self.bounding_box['top'] + margin
        self.yb_bound = self.bounding_box['top'] + self.bounding_box['height'] - margin
        
        self.sc = mss()
        self.last_score = 0
        self.done = False
        self.score_buffer = deque(maxlen=100)
        
    def step(self, action): # action 
        
        self._apply_action(action)
        
        sct_img = self.sc.grab(self.bounding_box)
        frame = np.array(sct_img)[:, :, 0:3]
        
        # recognizes that the retry screen has appeared, error catch for done parameters 1 and 2
        if (sum(sum(sum(frame == self.done_screen))) >= 1300000):
            self.done = True
        
        # in osu, if score != 0 and does not change for some time (no hits), health decays rapidly
        # can use this functionality as a 'done' parameter
        if (len(self.score_buffer) == self.score_buffer.maxlen) and (len(set(self.score_buffer)) == 1 and self.score_buffer[0] != 0):
            self.done = True
        
        score = frame[13:40, 650:800, :]
        score = int(get_score(score, self.hardcode_list))
        
        # resize 
        state = self._downscale(frame)
        
        # error catch for when the score is noticeably miscalcualted
        if (len(str(score)) - len(str(self.last_score)) > 2 or self.last_score > score): 
            score = self.last_score
        
        self.score_buffer.append(score)
        reward = score - self.last_score
        
        self.last_score = score
        
        return state, reward, self.done
    
    def reset(self):
        self.score_buffer.clear()
        self.last_score = 0
        self.done = False
        
        time.sleep(5) # wait for death animation to finish and reset screen to pop up
        
        
        pydirectinput.moveTo(990, 535)
        time.sleep(1)
        leftClick() # win32api, pyautogui doesnt work for click
        
        time.sleep(1) # wait for beatmap to load in 
        
        # current frame
        self.current_frame = np.array(self.sc.grab(self.bounding_box))[:, :, 0:3]
        state = self._downscale(self.current_frame)
        return state
    
    def _apply_action(self, action):
        # conquer relax mode first
        # actions are either moveTo(x, y), or do nothing
        
        if action is None: # do nothing
            return 
        
        x, y = action
        pydirectinput.moveTo(x, y)
        # leftClick()
        
    def _downscale(self, frame):
        '''
        downscale frame for memory/computation efficiency
        '''
        state = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dsize=(80, 60))
        return state
    
    def _get_random_action(self, margin=30):
        if np.random.rand() > 0.5:
            action = np.random.randint(self.xl_bound, self.xr_bound), np.random.randint(self.yt_bound, self.yb_bound)
        else:
            action = None
        return action
        
        
        
        