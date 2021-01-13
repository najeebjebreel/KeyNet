import torch
from torch.utils import data
class Dataset(data.Dataset):
    
    def __init__(self, data_set, signed = False):
        self.data_set = data_set
        self.signed = signed
        
    def __getitem__(self, index):
        
        if self.signed:
            x = self.data_set[0][index]
            y = self.data_set[1][index]
            f = 1
        else:
            x = self.data_set[0][index]
            y = self.data_set[1][index]
            f = 0

        return x, y, f
    
    def __len__(self):
        if self.signed:
            return len(self.data_set[0])
        return len(self.data_set[0])
    
class BBoxDtaset(data.Dataset):
    
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
 
    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
           
    
        return x, y
    
    def __len__(self):
        return len(self.inputs)
      
      