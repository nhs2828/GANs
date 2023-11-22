import torch
import torch.nn as nn
""" Extracted from the article
Architecture guidelines for stable Deep Convolutional GANs
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
• Use batchnorm in both the generator and the discriminator.
• Remove fully connected hidden layers for deeper architectures.
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.
• Use LeakyReLU activation in the discriminator for all layers.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, img_dim, features_gen):
        super().__init__()
        self.seq = nn.Sequential(
            # Input: Batch x channels_noise x 1 x 1
            # projection and reshape
            # img_size = 4x4
            nn.ConvTranspose2d(
                in_channels = noise_size,
                out_channels = 1024,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias=False,
            ),nn.BatchNorm2d(features_gen*16), nn.ReLU(),
            # CONV 1
            # img_size = 8x8
            nn.ConvTranspose2d(
                in_channels = 1024,
                out_channels = 512,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False,
            ),nn.BatchNorm2d(512), nn.ReLU(),
            # CONV 2
            # img_size = 16x16
            nn.ConvTranspose2d(
                in_channels = 512,
                out_channels = 256,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False,
            ),nn.BatchNorm2d(256),nn.ReLU(),
            # CONV 3
            # img_size = 32x32
            nn.ConvTranspose2d(
                in_channels = 256,
                out_channels = 128,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False,
            ),nn.BatchNorm2d(features_gen*2), nn.ReLU(),
            # CONV 4
            nn.ConvTranspose2d(in_channels=128, out_channels=img_dim, kernel_size=4, stride=2, padding=1),
            # Output: Batch x img_dim x 64 x 64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.seq(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim, features_dis):
        super().__init__()
        self.seq = nn.Sequential(
            # batch x img_dim x 64 x 64
            nn.Conv2d(img_dim, features_dis, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32 x 32
            nn.Conv2d(
                in_channels = features_dis,
                out_channels = features_dis*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features_dis*2),
            nn.LeakyReLU(0.2),
            # 16 x 16
            nn.Conv2d(
                in_channels = features_dis*2,
                out_channels = features_dis*4,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False,
            ),
            nn.BatchNorm2d(features_dis*4),
            nn.LeakyReLU(0.2),
            # 8 x 8
            nn.Conv2d(
                in_channels = features_dis*4,
                out_channels = features_dis*8,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False,
            ),
            nn.BatchNorm2d(features_dis*8),
            nn.LeakyReLU(0.2),
            # 4 x 4
            nn.Conv2d(features_dis*8, 1, kernel_size=4, stride=2, padding=0),
            # out: 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(x)


#  All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02
def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)