import numpy as np
import math
from posu.emulator_utils import *


class Circle:
    def __init__(self, center, startTime, radius=10, approachRate=1000):
        self.center = center
        self.radius = radius
        self.startTime = startTime
        self.approachRate = approachRate
        self.appear = startTime - approachRate
        self.disappear = startTime + approachRate // 4
        
        self.hit = False
        self.type = 'circle'
    
    def draw(self, screen, time, stack_offset=0):
        if self.hit: return
        if not self.appear <= time <= self.disappear: return
        
        # get fade in/out opacity
        if time < self.startTime:
            opacity = calc_fade_in(time, self.appear, self.startTime)
        else:
            opacity = calc_fade_out(time, self.startTime, self.disappear)
        
        # handle offset
        center = self.center
        if stack_offset:
            center = (self.center[0] - 2 * stack_offset, self.center[1] - 2 * stack_offset)
        
        # draw hit circle
        draw_filled_circle(screen, center, self.radius, opacity)
    
    def draw_approach_circle(self, screen, time, stack_offset=0):
        if self.hit: return
        if not self.appear <= time <= self.startTime: return
        
        # get fade in opacity
        opacity = calc_fade_in(time, self.appear, self.startTime, limit=0.5)
        
        # handle offset
        center = self.center
        if stack_offset:
            center = (self.center[0] - 2 * stack_offset, self.center[1] - 2 * stack_offset)
        
        # draw approach circle
        radius = int(self.radius + 3 * (self.startTime / (self.startTime - self.appear) - (1 / (self.startTime - self.appear)) * time) * self.radius)
        draw_empty_circle(screen, center, radius, opacity)
        
        
class Slider:
    def __init__(self, startTime, endTime, points, repeatCount, radius=10, approach=1000):
        self.startTime = startTime
        self.endTime = endTime
        self.appear = startTime - approach
        self.disappear = endTime + approach // 4
        self.points = points
        self.numPoints = len(points)
        self.repeatCount = repeatCount
        self.radius = radius
        self.center = points[0]
        self.type = 'slider'
    
    def draw(self, screen, time, stack_offset=0):
        # get circle/slider opacity
        if time < self.startTime:
            circle_opacity = calc_fade_in(time, self.appear, self.startTime)
        elif time <= self.endTime:
            circle_opacity = 1
        else:
            circle_opacity = calc_fade_out(time, self.endTime, self.disappear)
        slider_opacity = circle_opacity * 0.2

        # draw slider body
        draw_slider_body(screen, self.points, self.radius, slider_opacity)
        
        # keep the circle in the start/ending position
        if time < self.startTime:
            draw_filled_circle(screen, self.points[0], self.radius, circle_opacity)
        if time > self.endTime:
            end_point = self.points[0] if self.repeatCount % 2 == 0 else self.points[-1]
            draw_filled_circle(screen, end_point, self.radius, circle_opacity)
        
        # draw sliding hit circle
        if self.startTime <= time <= self.endTime:
            one = (self.endTime - self.startTime) / self.repeatCount 
            rep_num = (time - self.startTime) // one
            time_within_rep = (time - self.startTime) % one
            t = time_within_rep / one
            if rep_num % 2 != 0:
                t = 1 - t
            cx = int(sum([self.points[i][0] * (1 - t) ** (self.numPoints - i - 1) * t ** (i) * math.comb(self.numPoints - 1, i) for i in range(self.numPoints)]))
            cy = int(sum([self.points[i][1] * (1 - t) ** (self.numPoints - i - 1) * t ** (i) * math.comb(self.numPoints - 1, i) for i in range(self.numPoints)]))
            draw_filled_circle(screen, (cx, cy), self.radius, circle_opacity)

    def draw_approach_circle(self, screen, time, stack_offset=0):
        if not self.appear <= time <= self.startTime: return
        
        # get fade in opacity
        opacity = calc_fade_in(time, self.appear, self.startTime, limit=0.5)
        
        # draw approach circle
        radius = int(self.radius + 3 * (self.startTime / (self.startTime - self.appear) - (1 / (self.startTime - self.appear)) * time) * self.radius)
        draw_empty_circle(screen, self.points[0], radius, opacity)
        
        
class Stack:
    def __init__(self, hitObjects):
        self.hitObjects = sorted(hitObjects, key=lambda x: x.startTime, reverse=True)
        self.appear = hitObjects[0].appear
        self.startTime = hitObjects[0].startTime
        self.disappear = hitObjects[-1].disappear
        self.type = 'stack'
        
    def draw(self, screen, time):
        if not self.hitObjects: return
        for i, obj in enumerate(self.hitObjects):
            if obj.appear <= time <= obj.disappear:
                obj.draw(screen, time, stack_offset=i)
    
    def draw_approach_circle(self, screen, time):
        if not self.hitObjects: return
        for i, obj in enumerate(self.hitObjects):
            if obj.appear <= time <= obj.disappear:
                obj.draw_approach_circle(screen, time, stack_offset=i)
                
                
class Frame:
    def __init__(self, hitObjects):
        self.hitObjects = sorted(hitObjects, key=lambda x: x.startTime, reverse=True)

    def draw(self, screen, time):
        if not self.hitObjects: return
        for obj in self.hitObjects:
            obj.draw(screen, time)
        for obj in self.hitObjects: # draw approach circles on top of everything
            obj.draw_approach_circle(screen, time)