import torch
import torch.nn as nn
import torchvision.models as models


class Net(torch.nn.Module):
    def __init__(self ):
        super(Net, self).__init__()
        in_features = out_features = 35 # csv train data column count
        self.linear = nn.Linear( in_features, out_features )
        self._initialize_weights()
    pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            pass
        pass
    pass

    def forward(self, x):
        h = x
        h = self.linear(h)

        return h
    pass

pass