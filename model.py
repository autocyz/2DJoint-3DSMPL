import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.inputLinear = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.block = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.outputLinear = nn.Sequential(nn.Linear(1024, output_size)
        )
        self.init_layer(self.inputLinear)
        self.init_layer(self.block)
        self.init_layer(self.outputLinear)

    def init_layer(self, net):
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.inputLinear(x)
        x1 = self.block(x)
        x1 += x
        out = self.outputLinear(x1)
        return out
