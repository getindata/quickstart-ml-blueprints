"""Collection of encoder architectures"""
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleEncoder(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size[0]*self.image_size[1]*self.num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x


class SimpleConvEncoder(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.conv_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, stride=(1, 1), kernel_size=(5, 5)),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3)),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3)),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3))
            )
        
        self.flatten = nn.Flatten()
        self.lin_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, self.embedding_size)
        )
    
    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, self.num_channels, self.image_size[0], self.image_size[1])
        x = self.conv_encoder(x)
        x = self.flatten(x)
        x = self.lin_encoder(x)
        return x


## COPIED FROM https://gitlab.com/getindata/internal/aa_team_kagle_hm/-/blob/main/src/vertex_ai_plugin_demo/image_embeddings/model/encoders.py
class EncoderLinearLin1024(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size[0]*self.image_size[1]*3, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, self.embedding_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        return x


class EncoderConvBase(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(262144, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x


class EncoderConvLin1024(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(262144, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x


class EncoderConvCompr(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        super().__init__()
        self.image_size = [128, 128]
        self.embedding_size = embedding_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x


class EncoderConvCompr3Layer(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x
    

class EncoderConvCompr3LayerV2(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x
    

class EncoderConvCompr3LayerV3(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x
    

class EncoderConvLarger1stKernel(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7200, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        return x


class EncoderConvLarger2Kernels(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7200, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    

class EncoderConvLarger2KernelsLeaky(nn.Module):
    def __init__(self, embedding_size: int = 32, image_size: List[int] = [128, 128]):
        super().__init__()
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.num_channels = 3

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7200, self.embedding_size)
        )

    def forward(self, x):
        BS = x.shape[0]
        x = x.view(BS, 3, self.image_size[0], self.image_size[1])
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
