import torch
import torchvision
from torchvision import transforms

from torch import nn
from torchvision.ops import Conv2dNormActivation



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, layers=2, **kwargs):
        super().__init__()
        self.layers = layers
        self.conv1 = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU,
            **kwargs
            )
        self.conv2 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU,
            **kwargs
            )
        self.conv3 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU,
            **kwargs
            )
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        if self.layers == 2:
            return x
        return self.conv3(x)


class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # C x  H  x  W
        # 3 x 320 x 512 
        self.enc_conv1 = ConvBlock(in_channels=3, out_channels=64)
        # 64 x 320 x 512
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64 x 160 x 256

        self.enc_conv2 = ConvBlock(in_channels=64, out_channels=128)
        # 128 x 160 x 256
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128 x 80 x 128

        self.enc_conv3 = ConvBlock(in_channels=128, out_channels=256)
        # 256 x 80 x 128
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 256 x 40 x 128

        self.enc_conv4 = ConvBlock(in_channels=256, out_channels=512)
        # 512 x 40 x 128
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512 x 20 x 64

        self.enc_conv5 = ConvBlock(in_channels=512, out_channels=1024)
        # 1024 x 20 x 64      

        self.upconv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # 512 x 40 x 128
        # Concatenation from enc_conv4 produces # 1024 x 40 x 128
        self.dec_conv5 = ConvBlock(in_channels=1024, out_channels=512)
        # 512 x 40 x 128

        self.upconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # 256 x 80 x 256
        # Concatenation from enc_conv3 produces # 512 x 80 x 256
        self.dec_conv4 = ConvBlock(in_channels=512, out_channels=256)
        # 256 x 80 x 256

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # 128 x 160 x 512
        # Concatenation from enc_conv2 produces # 256 x 160 x 512
        self.dec_conv3 = ConvBlock(in_channels=256, out_channels=128)
        # 128 x 160 x 512

        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # 64 x 320 x 512
        # Concatenation from enc_conv1 produces # 128 x 320 x 512
        self.dec_conv2 = ConvBlock(in_channels=128, out_channels=64)
        # 64 x 320 x 512

        self.output_layer = ConvBlock(in_channels=64, out_channels=n_classes, kernel_size=1)
        # 3 x 320 x 512
    
    def forward(self, x):
        x_ec1 = self.enc_conv1(x)
        x = self.mp1(x_ec1)

        x_ec2 = self.enc_conv2(x)
        x = self.mp2(x_ec2)

        x_ec3 = self.enc_conv3(x)
        x = self.mp3(x_ec3)

        x_ec4 = self.enc_conv4(x)
        x = self.mp4(x_ec4)

        x = self.enc_conv5(x)

        x = self.upconv5(x)
        x = self.dec_conv5(torch.cat([x_ec4, x], dim=1))

        x = self.upconv4(x)
        x = self.dec_conv4(torch.cat([x_ec3, x], dim=1))

        x = self.upconv3(x)
        x = self.dec_conv3(torch.cat([x_ec2, x], dim=1))

        x = self.upconv2(x)
        x = self.dec_conv2(torch.cat([x_ec1, x], dim=1))

        x = self.output_layer(x)
        return x


class SegNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # C x  H  x  W
        # 3 x 320 x 512 
        self.enc_conv1 = ConvBlock(in_channels=3, out_channels=64)
        # 64 x 320 x 512
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 64 x 160 x 256

        self.enc_conv2 = ConvBlock(in_channels=64, out_channels=128)
        # 128 x 160 x 256
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 128 x 80 x 128

        self.enc_conv3 = ConvBlock(in_channels=128, out_channels=256, layers=3)
        # 256 x 80 x 128
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 256 x 40 x 64

        self.enc_conv4 = ConvBlock(in_channels=256, out_channels=512, layers=3)
        # 512 x 40 x 64
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 512 x 20 x 32

        self.enc_conv5 = ConvBlock(in_channels=512, out_channels=1024, layers=3)
        # 1024 x 20 x 32
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 1024 x 10 x               
        #### ^^ENCODER^^ | DECODER

        self.upconv5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 1024 x 20 x 32
        self.dec_conv5 = ConvBlock(in_channels=1024, out_channels=512, layers=3)
        # 512 x 20 x 32

        self.upconv4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 512 x 40 x 64
        self.dec_conv4 = ConvBlock(in_channels=512, out_channels=256, layers=3)
        # 256 x 40 x 64

        self.upconv3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 256 x 80 x 128
        self.dec_conv3 = ConvBlock(in_channels=256, out_channels=128, layers=3)
        # 128 x 80 x 128

        self.upconv2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 128 x 160 x 256
        self.dec_conv2 = ConvBlock(in_channels=128, out_channels=64)
        # 64 x 160 x 256

        self.upconv1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 64 x 320 x 512
        self.dec_conv1 = ConvBlock(in_channels=64, out_channels=n_classes)
        # 3 x 320 x 512

    
    def forward(self, x):
        # Encode
        x_ec1 = self.enc_conv1(x)
        shape1 = x_ec1.shape
        x, idx1 = self.mp1(x_ec1)

        x_ec2 = self.enc_conv2(x)
        shape2 = x_ec2.shape
        x, idx2 = self.mp2(x_ec2)

        x_ec3 = self.enc_conv3(x)
        shape3 = x_ec3.shape
        x, idx3 = self.mp3(x_ec3)

        x_ec4 = self.enc_conv4(x)
        shape4 = x_ec4.shape
        x, idx4 = self.mp4(x_ec4)

        x_ec5 = self.enc_conv5(x)
        shape5 = x_ec5.shape
        x, idx5 = self.mp5(x_ec5)

        # Decode
        x = self.upconv5(x, indices=idx5, output_size=shape5)
        x = self.dec_conv5(x)

        x = self.upconv4(x, indices=idx4, output_size=shape4)
        x = self.dec_conv4(x)

        x = self.upconv3(x, indices=idx3, output_size=shape3)
        x = self.dec_conv3(x)

        x = self.upconv2(x, indices=idx2, output_size=shape2)
        x = self.dec_conv2(x)

        x = self.upconv1(x, indices=idx1, output_size=shape1)
        x = self.dec_conv1(x)

        x = nn.Softmax(dim=1)(x)
        return x

data_transforms = transforms.Compose([
    transforms.Resize([320, 512], interpolation=torchvision.transforms.InterpolationMode.NEAREST),
])

def decode_mask(mask_1d):
    red = torch.where(mask_1d == 1, 255, 0)
    green = torch.where(mask_1d == 2, 255, 0)
    blue = torch.zeros_like(red)
    decoded_mask = torch.stack([red, green, blue])
    return decoded_mask
