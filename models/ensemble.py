import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

class EnsembleTwo(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleTwo, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # Pass the input through both models
        output1 = self.model1(x)
        output2 = self.model2(x)

        output = (output1 + output2)/2
        return output

class EnsembleFour(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super(EnsembleFour, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, x):
        # Pass the input through both models
        output1 = self.model1(x)
        output2 = self.model2(x)
        output3 = self.model3(x)
        output4 = self.model4(x)

        output = (output1 + output2 + output3 + output4)/4

        return output