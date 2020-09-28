import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, ContinuousBernoulli
import numpy as np


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def forward(self, x=None):

        bs = x.size(0)
        ls = self.encoder.latent_dims

        mu, sigma = self.encoder(x)
        self.pz = Independent(
            Normal(loc=torch.zeros(bs, ls).to(self.device), scale=torch.ones(bs, ls).to(self.device)),
            reinterpreted_batch_ndims=1)
        self.qz_x = Independent(Normal(loc=mu, scale=torch.exp(sigma)), reinterpreted_batch_ndims=1)

        self.z = self.qz_x.rsample()
        decoded = self.decoder(self.z)

        return decoded


    def compute_loss(self, x, y, scale_kl=False):

        px_z = Independent(ContinuousBernoulli(logits=y), reinterpreted_batch_ndims=3)

        px = px_z.log_prob(x)
        kl = self.pz.log_prob(self.z) - self.qz_x.log_prob(self.z)

        if scale_kl:
            kl = kl * scale_kl

        loss = -(px + kl).mean()

        return loss, kl.mean().item(), px.mean().item()

    def rmse(self, input, target):
        return torch.sqrt(F.mse_loss(input, target))

class ResLayer(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, norm_layer=None, padding=1):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.c0 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=stride, dilation=dilation, padding=padding)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=stride, dilation=1, padding=padding)
        self.bn1 = norm_layer(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, dilation=1, padding=1)
        self.bn2 = norm_layer(channels)
        self.stride = stride

    def forward(self, x):

        identity = self.c0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResBlock(nn.Sequential):
    def __init__(self, n_layers, inchannels, outchannels, downsampling='stride', layer=ResLayer, padding=1):

        if downsampling == 'stride': stride = 2; dilation = 1
        elif downsampling == 'dilation': stride = 1; dilation = 2
        else: stride = 1; dilation = 1

        layers = [
            layer(inchannels, outchannels, stride, dilation, padding=padding)
        ]
        for i in range(1, n_layers):
            layers.append(layer(outchannels, outchannels, stride=1, dilation=1))

        super().__init__(*layers)

class DenseBlock(nn.Module):
    def __init__(self, n_layers, inchannels, outchannels, downsampling='stride', layer=ResLayer):

        super().__init__()

        if downsampling == 'stride': stride = 2; dilation = 1
        elif downsampling == 'dilation': stride = 1; dilation = 2
        else: stride = 1; dilation = 1

        self.layers = nn.ModuleList([
            layer(inchannels, outchannels, stride, dilation)
        ])

        self.convlayers = nn.ModuleList([])
        self.bnlayers = nn.ModuleList([])

        for i in range(1, n_layers):
            self.layers.append(layer(outchannels, outchannels, stride=1, dilation=1))
            self.convlayers.append(nn.Conv2d(outchannels*(i+1), outchannels, 3, 1, 1))
            self.bnlayers.append(nn.BatchNorm2d(outchannels))

    def forward(self, x):
        activations = []
        for n, layer in enumerate(self.layers):
            x = layer(x)
            activations.append(x)
            if n > 0:
                # this could get huge, check does this copy the weights or reference?
                x = torch.cat(activations, 1)
                x = self.convlayers[n-1](x)
                x = self.bnlayers[n-1](x)
                x = F.leaky_relu(x)
                # x = x + identity
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dims=64, in_channels=3, block=ResBlock):
        super().__init__()

        self.latent_dims = latent_dims

        # TODO fix for different downscale factors and for dilation or stride or pool
        strides = [2, 2, 2, 1, 1]

        block_sizes = [3, 3, 4, 2]

        if self.latent_dims == 256:
            ds = None
        else:
            ds = 'stride'

        self.l0 = ResLayer(in_channels, 32, stride=strides[0])
        self.l1 = block(block_sizes[0], 32, 64)
        self.l2 = block(block_sizes[1], 64, 128)
        self.l3 = block(block_sizes[2], 128, 256, downsampling=ds)
        self.l4 = block(block_sizes[3], 256, 512, downsampling=None)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.mu = nn.Linear(512, self.latent_dims)
        self.sigma = nn.Linear(512, self.latent_dims)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.l0(x)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

class FeatureEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.2),
        )

        self.mu = nn.Linear(256, latent_dim)
        self.sigma = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims=64, out_channels=3, num_featues=None, block=ResBlock):
        super().__init__()

        assert latent_dims in [64, 144, 256]
        pd = [3,3,3,2,2] if latent_dims == 144 else [1]*5


        self.latent_dims = latent_dims

        block_sizes = [3, 3, 4, 2]

        self.l0 = ResLayer(1, 512, stride=1, padding=pd[0])
        self.l1 = block(block_sizes[0], 512, 256, downsampling=None, padding=pd[1])
        self.l2 = block(block_sizes[1], 256, 128, downsampling=None, padding=pd[2])
        self.l3 = block(block_sizes[2], 128, 64, downsampling=None, padding=pd[3])
        self.l4 = block(block_sizes[3], 64, 32, downsampling=None, padding=pd[4])

        self.c2 = nn.Conv2d(32, out_channels, 5, 1, 2)

        self.c0 = None
        if num_featues is not None:
            self.c0 = nn.Conv2d(32, 1, 3, 1, 1)
            self.bn1 = nn.BatchNorm1d(128*128)
            self.fc1 = nn.Linear(128*128, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc2 = nn.Linear(1024, num_featues)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.latent_dims == 256:
            shp = 16
        elif self.latent_dims == 64:
            shp = 8
        else:
            shp = 12

        x = x.view(x.size(0), 1, shp, shp)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.l0(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.l1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.l2(x)
        if self.latent_dims == 64:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.l3(x)
        x = self.l4(x)

        img = self.c2(x)

        if self.c0 is not None:
            f = self.c0(x)
            f = torch.flatten(f, start_dim=1)
            f = F.leaky_relu(self.bn1(f))
            f = F.leaky_relu(self.bn2(self.fc1(f)))
            features = self.fc2(f)

            return img, features

        return img
