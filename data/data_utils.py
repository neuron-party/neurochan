import torch
import torch.nn as nn
import numpy as np
import cv2


def create_bounding_box_xyxy(center, radius):
    tl_x, tl_y = center[0] - radius, center[1] - radius
    br_x, br_y = center[0] + radius, center[1] + radius
    return tl_x, tl_y, br_x, br_y


def visualize_dataloader_od(dataloader):
    x, y = next(iter(dataloader))
    img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().copy()
    bbs = y['boxes'][0]
    img = cv2.rectangle(
        img,
        (int(bbs[0]), int(bbs[1])),
        (int(bbs[2]), int(bbs[3])),
        (255, 0, 0),
        1
    )
    return img