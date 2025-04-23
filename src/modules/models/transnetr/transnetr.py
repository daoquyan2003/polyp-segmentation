import torch
import torch.nn as nn

from .resnet import resnet50


class Conv2D(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        bias=True,
        act=True,
    ):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act is True:
            x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        s = self.shortcut(inputs)
        return self.relu(x + s)


class residual_transformer_block(nn.Module):
    def __init__(self, in_c, out_c, patch_size=4, num_heads=4, num_layers=2):
        super().__init__()

        self.ps = patch_size
        self.c1 = Conv2D(in_c, out_c)

        self.dim = (patch_size**2) * out_c  # Calculate d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=num_heads
        )
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.c2 = Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.c3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.r1 = residual_block(out_c, out_c)

    def forward(self, inputs):
        x = self.c1(inputs)

        b, c, h, w = x.shape

        # Check divisibility by patch size
        assert (
            h % self.ps == 0 and w % self.ps == 0
        ), f"Input dimensions {h}x{w} must be divisible by patch size {self.ps}"

        # Split into patches and flatten
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # (b, c, h/p, w/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (b, h/p, w/p, c, p, p)
        x = x.view(b, -1, self.dim)  # (b, num_patches, dim)

        # Transformer requires (seq_len, batch, dim)
        x = x.permute(1, 0, 2)  # (num_patches, b, dim)
        x = self.te(x)
        x = x.permute(1, 0, 2)  # (b, num_patches, dim)

        # Reshape back to spatial structure
        x = x.view(b, h // self.ps, w // self.ps, c, self.ps, self.ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, h, w)  # Restore original shape

        x = self.c2(x)
        s = self.c3(inputs)
        x = self.relu(x + s)
        x = self.r1(x)
        return x


class TransNetR(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        """Encoder."""
        backbone = resnet50()
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.e1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.e2 = Conv2D(256, 64, kernel_size=1, padding=0)
        self.e3 = Conv2D(512, 64, kernel_size=1, padding=0)
        self.e4 = Conv2D(1024, 64, kernel_size=1, padding=0)
        """Decoder."""
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.r1 = residual_transformer_block(64 + 64, 64)
        self.r2 = residual_transformer_block(64 + 64, 64)
        self.r3 = residual_block(64 + 64, 64)
        """Classifier."""
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder."""
        x0 = inputs
        x1 = self.layer0(x0)  # [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  # [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  # [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  # [-1, 1024, h/16, w/16]
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        e1 = self.e1(x1)
        e2 = self.e2(x2)
        e3 = self.e3(x3)
        e4 = self.e4(x4)
        """Decoder."""
        x = self.up(e4)
        x = torch.cat([x, e3], axis=1)
        x = self.r1(x)

        x = self.up(x)
        x = torch.cat([x, e2], axis=1)
        x = self.r2(x)

        x = self.up(x)
        x = torch.cat([x, e1], axis=1)
        x = self.r3(x)

        x = self.up(x)
        """Classifier."""
        outputs = self.outputs(x)
        return outputs


if __name__ == "__main__":
    x = torch.randn((4, 3, 384, 384))
    model = TransNetR()
    y = model(x)
    print(y.shape)
