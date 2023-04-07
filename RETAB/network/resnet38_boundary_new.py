import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d
from tool import pyutils

class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        #self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)
        self.f9 = torch.nn.Conv2d(448, 1, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

    def forward(self, x):
        H, W = x.size()[2:]
        d = super().forward_as_dict(x)
        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1))
        x = F.interpolate(x, (H,W), mode='bilinear', align_corners=True)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

