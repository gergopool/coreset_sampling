import torch.nn as nn

__all__ = ['prjection']

class Projection(nn.Module):

    def __init__(self, num_channels_in, num_channels_out, num_hidden=256):
        super(Projection, self).__init__()
        self.representation = nn.Linear(num_channels_in, num_hidden)
        self.projection = nn.Sequential(
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_channels_out),
            nn.BatchNorm1d(num_channels_out),
            nn.ReLU()
        )

    def forward(self, inputs):
        z = self.representation(inputs)
        out = self.projection(z)
        return z, out

def projection(num_channels_in, num_channels_out):
    return Projection(num_channels_in, num_channels_out)