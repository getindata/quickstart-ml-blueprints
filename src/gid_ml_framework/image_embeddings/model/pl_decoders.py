"""Collection of decoders architectures"""
from torch import nn 

class SimpleDecoder(nn.Module):
    def __init__(self, embedding_size: int = 32):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = [128, 128]
        self.num_channels = 3

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_size[0]*self.image_size[1]*self.num_channels)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
