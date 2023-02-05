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


def main():
    parser = BeatmapParser()
    parser.parseFile('leaf.osu')
    beatmap_d = parser.build_beatmap()
    
    hitObjects = FormatParsedBeatmap(beatmap_d)
    images, labels = [], []
    
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
    i = 0
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

        label = None
        for obj in objs:
            if obj.type == 'stack':
                for c in obj.hitObjects:
                    if c.startTime == time:
                        label = [c.center, c.radius]
            else:
                if obj.startTime == time:
                    label = [obj.center, obj.radius]
        if label:
            img = pygame.surfarray.array3d(screen).transpose(2, 0, 1)
            images.append(img)
            labels.append(label)

        pygame.display.flip()
        pygame.display.update()

        i += 1
        if i < len(hitObjects):
            time = hitObjects[i].startTime
        else:
            break
    
    pygame.quit()

    dataset = OsuDataset(images[1:], labels[1:])
    trainloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)
    
    print('finished creating dataset')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    device = torch.device('cuda:0')
    model = model.to(device)
    train, val = torch.utils.data.random_split(dataset, [667, 75])
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=1)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss, val_loss = [], []
    print('about to train')
    for e in range(100):
        tl, vl = [], []
        for x, y in trainloader:
            x = [x.squeeze(0).to(device).float()]
            y = [{
                'labels': torch.ones(1, dtype=torch.int64).to(device),
                'boxes': y['boxes'].to(device)
            }]
            optimizer.zero_grad()
            loss_dict = model(x, y)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            tl.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for x, y in valloader:
                x = [x.squeeze(0).to(device).float()]
                y = [{
                    'labels': torch.ones(1, dtype=torch.int64).to(device),
                    'boxes': y['boxes'].to(device)
                }]
                loss_dict = model(x, y)
                loss = sum(loss_dict.values())

                vl.append(loss.detach().cpu().numpy())

        train_loss.append(np.mean(tl))
        val_loss.append(np.mean(vl))
        
        print(f'TL: {np.mean(train_loss)}, VL: {np.mean(val_loss)}')
        
if __name__ == '__main__':
    main()