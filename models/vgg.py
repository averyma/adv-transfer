'''https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/models/vgg.py'''

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, dataset, num_classes=10, input_normalization=True, enable_batchnorm=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], enable_batchnorm)

        if dataset in ['cifar10', 'cifar100', 'svhn']:
            self.classifier = nn.Linear(512, num_classes)

        elif dataset in ['tiny', 'dtd']:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, num_classes),
            )

        elif dataset in ['imagenet', 'imagenette']:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                # nn.Dropout(global_params.dropout_rate),
                nn.Dropout(0.2),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                # nn.Dropout(global_params.dropout_rate),
                nn.Dropout(0.2),
                nn.Linear(4096, num_classes),
            )

        self.input_normalization = input_normalization

    def per_image_standardization(self, x):
        """
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
        """
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1,2,3), keepdim = True)
        stddev = torch.std(x, dim=(1,2,3), keepdim = True)
        adjusted_stddev = torch.max(stddev, (1./np.sqrt(_dim)) * torch.ones_like(stddev))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        if self.input_normalization:
            x = self.per_image_standardization(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, enable_batchnorm):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if enable_batchnorm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
