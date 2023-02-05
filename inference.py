# generate data
import pygame
from pygame.locals import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt

from posu.env_utils import *
from posu.emulator import *
from posu.emulator_utils import *
from osuparser.beatmapparser import *
from data.data_utils import *
from data.dataset import *
from utils.export_video import *



def main():
    parser = BeatmapParser()
    parser.parseFile('leaf.osu')
    beatmap_d = parser.build_beatmap()
    hitObjects = FormatParsedBeatmap(beatmap_d)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    device = torch.device('cuda:0')
    weights = torch.load('../neurosama/neurosama/posu/hit_circle_detector_1.pth')
    model.load_state_dict(weights['mode'])
    model = model.eval()
    model = model.to(device)

    # load model n shit
    frames = []

    # get frames
    pygame.quit()
    pygame.init()

    size = width, height = 192 + 80, 144 + 80
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(size, DOUBLEBUF, 16)
    screen.fill(BLACK)

    ms = pygame.Surface((10, 10))
    draw_cursor(ms, (5, 5), 5)
    cursor = pygame.cursors.Cursor((5, 5), ms)
    pygame.mouse.set_cursor(cursor)

    run = True
    time = 0
    last_x, last_y = 96, 72
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False

        screen.fill(BLACK)

        objs = []
        for obj in hitObjects:
            if obj.appear < time < obj.disappear:
                objs.append(obj)

        frame = Frame(objs)
        frame.draw(screen, time)

        # make model prediction
        img = pygame.surfarray.array3d(screen)[40: 232, 40: 184].transpose(2, 1, 0) / 255
        x = torch.Tensor(img).to(device)
        pred = model([x.float()])
        if len(pred[0]['boxes']):
            x1, y1, x2, y2 = pred[0]['boxes'][0]
            x, y = int((x1 + x2) // 2) + 40, int((y1 + y2) // 2) + 40
            last_x, last_y = x, y
        else:
            x, y = last_x, last_y

        # control mouse
        # pygame.mouse.set_pos((x, y))
        draw_cursor(screen, (x, y), 5)

        # get frame img
        frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        frames.append(frame)

        pygame.display.flip()
        pygame.display.update()

        time += 1000/60


    for i in range(len(frames)):
        frames[i] = frames[i].astype(np.uint8)
    export_video(frames, fps=60)
    

    
if __name__ == '__main__':
    main()