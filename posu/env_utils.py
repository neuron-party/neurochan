import numpy as np
import math
from posu.emulator import *
from posu.emulator_utils import *


def convert(hitObject):
    if hitObject['object_name'] == 'circle':
        x, y = hitObject['position']
        scaled_center = ((x * 192 // 512 + 40, y * 192 // 512 + 40))
        return Circle(scaled_center, hitObject['startTime'])
    elif hitObject['object_name'] == 'slider':
        scaled_points = []
        for x, y in hitObject['points']:
            scaled_points.append((x * 192 // 512 + 40, y * 192 // 512 + 40))
        return Slider(hitObject['startTime'], hitObject['end_time'], scaled_points, hitObject['repeatCount'])

def same_pos(unscaled, scaled):
    return unscaled[0] * 192 // 512 + 40 == scaled[0] and unscaled[1] * 192 // 512 + 40 == scaled[1]
    
def FormatParsedBeatmap(beatmap_d: dict):
    hitObjects = []
    stack = []
    count = 0
    for hitObject in beatmap_d['hitObjects']:
        
        if hitObject['object_name'] not in ['circle', 'slider']: continue
        
        if not stack and hitObjects and same_pos(hitObject['position'], hitObjects[-1].center):
            stack.append(hitObjects[-1])
            stack.append(convert(hitObject))
            hitObjects.pop()
            continue
        elif stack and same_pos(hitObject['position'], stack[-1].center):
            stack.append(convert(hitObject))
            continue
        if stack: 
            hitObjects.append(Stack(stack))
            stack = []
        count += 1
        hitObjects.append(convert(hitObject))
    return hitObjects