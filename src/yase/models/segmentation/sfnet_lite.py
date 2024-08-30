import torch
import torch.nn as nn
from sfnet.sfnet_lite import SFNetLiteModel

class SFNetLite(nn.Module):
    def __init__(self, num_classes=19):
        super(SFNetLite, self).__init__()
        self.model = SFNetLiteModel(num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
