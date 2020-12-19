import numpy as np
import torch
import torch.nn.functional as F

class MNet(torch.nn.Module):
    def __init__(self, d_feature):
        super(MNet, self).__init__()
        self.inprod = torch.nn.Linear(d_feature, 1,bias=False)
        self.weight_init()

    def forward(self, x):
        z = self.inprod(x)
        return z

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.uniform_(m.weight.data)
                #torch.nn.init.zeros_(m.weight.data)

