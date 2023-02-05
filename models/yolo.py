import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO(nn.Module):
    def __init__(self):
        # network blocks are built in the same fashion as Figure 3 (https://arxiv.org/pdf/1506.02640.pdf)
        # last 4 conv layers belong to YOLO
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.block4 = nn.Sequential(
            # repeat these 2 layers 4 times
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.block5 = nn.Sequential(
            # repeat these 2 layers 2 times
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(1024)
        
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 3 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 3 * 4 * (1 * 5 + 2)),
            nn.Sigmoid()
        )   
        
    def forward(self, x):
        '''
        Inputs: Tensor of shape [b, 3, 224, 224]
        Outputs: Tensor of shape [b, 1000]
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        x = self.fc(x)
        
        x = x.view(-1, 3, 4, 1 * 5 + 2)
        return x
    
    
class YoloLoss(nn.Module):
    '''
    Maybe remove the wh loss and have it do purely center regression i.e change the tensor shape to [b, 3, 4, 5] where indices 3 and 4 are x y
    For the task of osu, the wh predictions are useless
    '''
    def __init__(self, lambda_coord = 5, lambda_noobj = 0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, x, y):
        '''
        x, y: tensors of shape [b, 3, 4, 7] 
        '''
        batch_size = x.shape[0]
        
        obj_mask = y[:, :, :, 2] == 1 
        pred = x[obj_mask]
        targets = y[obj_mask]
        
        xy_loss = torch.sum(((pred[:, 3:5] - targets[:, 3:5]) ** 2), dim=1) # dim=0 reserved for batch size
        wh_loss = torch.sum(((torch.sqrt(pred[:, 5:7]) - torch.sqrt(targets[:, 5:7])) ** 2), dim=1)
        obj_loss = torch.sum(((pred[:, 2] - targets[:, 2]) ** 2))
        cls_loss = self.classification_loss(pred[:, 0:2], targets[:, 0:2])
        
        noobj_pred = x[~obj_mask]
        noobj_targets = y[~obj_mask]
        noobj_loss = torch.sum(((noobj_pred[:, 2] - noobj_targets[:, 2]) ** 2))
        
        total_loss = self.lambda_coord * (xy_loss + wh_loss) + obj_loss + self.lambda_noobj * noobj_loss + cls_loss
        return total_loss