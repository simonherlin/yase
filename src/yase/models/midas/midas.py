import torch
import torch.nn as nn
from midas.midas_net import MidasNet

class MiDaS(nn.Module):
    def __init__(self, model_path="model-f6b98070.pt"):
        super(MiDaS, self).__init__()
        self.model = MidasNet(model_path)
        
    def forward(self, x):
        return self.model(x)
