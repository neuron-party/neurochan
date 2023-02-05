import torch
import torch.nn as nn
import numpy as np
from data.data_utils import *


class OsuDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.data, self.labels = [], []
        for img, label in zip(images, labels):
            img = img.transpose(0, 2, 1)
            img = img / 255
            self.data.append(img)
            
            # center radius to xyxy
            tl_x, tl_y, br_x, br_y = create_bounding_box_xyxy(label[0], label[1])
        
            bbs = torch.Tensor([tl_x, tl_y, br_x, br_y])
            clss = torch.ones(1, dtype=torch.int64)
            
            lab = {'boxes': bbs, 'labels': clss}
            self.labels.append(lab)
                
        assert len(self.data) == len(self.labels)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.data)