import torch
import torch.nn as nn
import torch.nn.functional as F

""" The private model used with ResNet18 model
This private model takes the output of the ResNet18 marked model as input and 
outputs the position of the owner's signature on the input sample"""
class WMPrivate(nn.Module):
    def __init__(self):
        super(WMPrivate, self).__init__()
        self.wmlayer1 = nn.Linear(10, 20)
        self.wmlayer2 = nn.Linear(20, 10)
        self.wmlayer3 = nn.Linear(10, 6)
        
    def forward(self, x):
        x = self.wmlayer1(x)
        x = self.wmlayer2(x)
        x = self.wmlayer3(x)
        return  F.log_softmax(x, dim = 0)

        