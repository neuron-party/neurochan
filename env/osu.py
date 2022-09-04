import cv2
import gym
import time
import random
import pyautogui
import win32api
import numpy as np
from mss import mss
from collections import deque

from utils.scoring import *


class Osu(gym.Env):
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
        
        score = frame[0:30, 675:800, :]
        score = int(get_score_2(score, self.hardcode_list))
        
        # error catch for when the score is noticeably miscalcualted
        if (len(str(score)) - len(str(self.last_score)) > 2 or self.last_score > score): 
            score = self.last_score
        
        self.score_buffer.append(score)
        reward = score - self.last_score
        
        self.last_score = score
        
        return frame, score, self.done
    
    def reset(self):
        self.score_buffer.clear()
        self.last_score = 0
        self.done = False
        
        # 960, 535 - reset coordinate
        
        # need to click twice for some reason
        time.sleep(1)
        pyautogui.click(960, 535)
        time.sleep(1)
        pyautogui.click(960, 535)
        
        # time.sleep(n) ensures that screen capturing does not begin in the very early stages of the map,
        # where get_score() function struggles due to the dimness of the entire map as it is loading in
        time.sleep(3)
        
        # current frame
        self.current_frame = np.array(self.sc.grab(self.bounding_box))[:, :, 0:3]
        return self.current_frame
    
    def _apply_action(self, action):
        x, y = action # mouse click
        # pyautogui moveto/dragto functions don't work within an osu beatmap; clicks work though (anticheat????)
        # pyautogui.moveTo(x, y) # currently moveto doesnt work in osu, anticheat?
        # pyautogui.dragTo(x, y)
        win32api.SetCursorPos(action)
        time.sleep(0.1) # wait 0.1 seconds?
        pyautogui.click()