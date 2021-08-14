import torch.nn as nn

__all__ = ['prjection']

class Projection(nn.Module):

    def __init__(self, num_channels_in, num_channels_out):
        super(Projection, self).__init__()
        self.representation = nn.Linear(num_channels_in, 256)
        self.projection = nn.Sequential(
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(256, num_channels_out),
            nn.BatchNorm1d(),
            nn.ReLU()
        )

    def forward(self, inputs):
        z = self.representation(inputs)
        x = self.projection(z)
        return z, x

def projection(num_channels_in, num_channels_out):
    return Projection(num_channels_in, num_channels_out)