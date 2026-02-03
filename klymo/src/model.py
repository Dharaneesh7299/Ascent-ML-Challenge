import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class Generator(nn.Module):
    """
    ESRGAN-style Generator (RRDBNet)
    Upscale factor: 4
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, upscale=4):
        super(Generator, self).__init__()
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.upscale = upscale

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        
        # Upsample 2x
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # Upsample 2x (Total 4x)
        if self.upscale >= 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        # Additional block if 8x required (hacky for now, standard is 4x usually)
        if self.upscale == 8:
             fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

class Discriminator(nn.Module):
    """
    VGG-Style Discriminator for GAN Loss
    """
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1, bias=False), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 8, 100, 3, 1, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(100, 1, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        out = self.net(x)
        # Average pooling to handle any input size
        return torch.sigmoid(F.adaptive_avg_pool2d(out, 1).view(x.size(0), -1))
