import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.instance_norm1 = nn.InstanceNorm2d(channels)
        self.instance_norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.instance_norm1(self.conv1(x)))
        x = self.instance_norm2(self.conv2(x))
        return x + residual

class Generator(nn.Module):
    def __init__(self, input_channels=3, base_channels=16):
        super().__init__()
        self.encoders = nn.Sequential(
            nn.Conv2d(input_channels * 2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # state size. base_channels x W/2 x H/2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # state size. base_channels*2 x W/4 x H/4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # state size. base_channels*4 x W/8 x H/8
        )
        
        self.residual_blocks = nn.Sequential(*[ResidualBlock(base_channels * 4) for _ in range(4)])
        
        self.decoders = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # state size. base_channels*2 x W/4 x H/4
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # state size. base_channels x W/2 x H/2
            nn.ConvTranspose2d(base_channels, base_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            # state size. base_channels/2 x W x H
        )

        self.final = nn.Sequential(
            nn.Conv2d(base_channels//2, input_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x, m):
        combined = torch.cat([x, m], dim=1)
        out = self.encoders(combined)
        out = self.residual_blocks(out)
        out = self.decoders(out)
        out = self.final(out)
        
        # Ensure output matches input spatial dimensions
        assert out.shape[2:] == x.shape[2:], f"Output shape {out.shape} doesn't match input shape {x.shape}"
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_channels=16):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self.conv_block(input_channels, base_channels, normalize=False),
            self.conv_block(base_channels, base_channels * 2),
            self.conv_block(base_channels * 2, base_channels * 4),
            self.conv_block(base_channels * 4, base_channels * 8),
            # self.conv_block(base_channels * 8, base_channels * 16),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),  #16*16 or 4x4?
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.flatten(self.model(x), 1)
        x = self.model(x)
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2,
                 rate: int = 10000,
                 input_size = 512,
                 device = 'cuda'):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="mean")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.target_size = input_size
        self.device = device

    def get_components(self, x, no_loss=False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(self.device)
        c = self.model.get_learned_conditioning(self.condition)
        if no_loss:
            loss = 0
        else:
            loss = self.model(z, c)[0]
        return z, loss

    def pre_process(self, x, target_size):
        processed_x = torch.zeros([x.shape[0], x.shape[1], target_size, target_size]).to(self.device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x

    def forward(self, x, target_info=None, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        self.target_info = target_info if target_info is not None else self.target_info
        zx, loss_semantic = self.get_components(x, True)
        zy, _ = self.get_components(self.target_info, True)
        if self.mode != 1:
            _, loss_semantic = self.get_components(self.pre_process(x, self.target_size))
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 0:
            return - loss_semantic
        elif self.mode == 1:
            # L2 loss between the input and target image in latent space.
            return self.fn(zx, zy)
        else:
            return self.fn(zx, zy) - loss_semantic * self.rate


class DiffusionTargetModel(nn.Module):
    """
    A virtual model which computes the textural loss in forward function.
    """

    def __init__(self, 
                 model,
                 condition: str = "A photo",
                 target_info: str = None,
                 input_size = 512,
                 device = 'cuda'):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = self.custom_loss
        self.target_info = target_info
        self.target_size = input_size
        self.device = device

    def get_encodings(self, x, no_loss=False):
        """
        Compute the encoded information of the input.
        :return: encoded info of x
        """
        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(self.device)
        c = self.model.get_learned_conditioning(self.condition)
        
        if no_loss:
            loss = 0
        else:
            loss = self.model(z, c)[0]

        return z, loss
    
    def custom_loss(self, x, target_info):
        return 1 - F.cosine_similarity(x, target_info).mean()

    def forward(self, x, target_info=None, components=False):
        """
        Compute the textural loss.
        The textural loss shows the distance between the input image and target image in latent space.
        """
        self.target_info = target_info if target_info is not None else self.target_info
        zx, loss_sematic = self.get_encodings(x, True)
        zy, _ = self.get_encodings(self.target_info, True)
        if components:
            return self.fn(zx, zy), loss_sematic
        
        return self.fn(zx, zy)
        
        