import math

import torch
import torch.nn as nn


class Channel_Attention(nn.Module):
    def __init__(self, F_g, F_l, size):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.ConvTranspose2d(F_g, F_l, kernel_size=size, stride=size),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True),
        )

        self.W_g_1 = nn.Sequential(
            nn.Conv2d(F_l, int(F_l / 16), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(F_l / 16)),
            nn.ReLU(inplace=True),
        )

        self.W_x_1 = nn.Sequential(
            nn.Conv2d(F_l, int(F_l / 16), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(F_l / 16)),
            nn.ReLU(inplace=True),
        )

        self.W = nn.Sequential(
            nn.Conv2d(int(F_l / 16), F_l, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        w_g = self.W_g(g)

        # cross H x W
        avg_g = nn.AvgPool2d(kernel_size=(w_g.size(2), w_g.size(3)))(w_g)
        max_g = nn.MaxPool2d(kernel_size=(w_g.size(2), w_g.size(3)))(w_g)
        avg_g_n = self.W_g_1(avg_g)
        max_g_n = self.W_g_1(max_g)
        add_g = avg_g_n + max_g_n

        avg_x = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))(x)
        max_x = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))(x)
        avg_x_n = self.W_x_1(avg_x)
        max_x_n = self.W_x_1(max_x)
        add_x = avg_x_n + max_x_n
        add = add_g + add_x

        return self.W(add)


class Spatial_Attention(nn.Module):
    def __init__(self, F_g, F_l, size):
        super().__init__()
        self.F_l = F_l
        self.W_g = nn.Sequential(
            nn.ConvTranspose2d(F_g, F_l, kernel_size=size, stride=size),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True),
        )
        self.W_x_conv = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.W_g_conv = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        kernel_size_x = int((math.log2(1024 + 64 - self.F_l) + 1) // 2)
        kernel_size_x = (
            kernel_size_x if (kernel_size_x % 2) else (kernel_size_x + 1)
        )
        self.W_x_cus = nn.Sequential(
            nn.Conv2d(
                3,
                1,
                kernel_size=kernel_size_x,
                stride=1,
                padding=int(kernel_size_x // 2),
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.W_g_cus = nn.Sequential(
            nn.Conv2d(
                3,
                1,
                kernel_size=kernel_size_x,
                stride=1,
                padding=(kernel_size_x // 2),
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        w_g = self.W_g(g)
        w_g_conv = self.W_g_conv(w_g)
        cat_g = torch.cat(
            (
                torch.mean(w_g, dim=1).unsqueeze(1),
                torch.max(w_g, dim=1).values.unsqueeze(1),
                w_g_conv,
            ),
            dim=1,
        )
        w_x_conv = self.W_x_conv(x)
        cat_x = torch.cat(
            (
                torch.mean(x, dim=1).unsqueeze(1),
                torch.max(x, dim=1).values.unsqueeze(1),
                w_x_conv,
            ),
            dim=1,
        )

        out_g = self.W_g_cus(cat_g)
        out_x = self.W_x_cus(cat_x)

        return self.sigmoid(out_g + out_x)


class SCGate(nn.Module):
    def __init__(self, F_g, F_l, size):
        super().__init__()
        self.spatial_attention = Spatial_Attention(F_g, F_l, size)
        self.channel_attention = Channel_Attention(F_g, F_l, size)

    def forward(self, g, x):
        spatial_attention = self.spatial_attention(g, x)
        x1 = x * spatial_attention.expand_as(x)
        channel_attention = self.channel_attention(g, x1)

        return x1 * channel_attention.expand_as(x1)


class Up2Convolution3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                ch_in, ch_out, kernel_size=2, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.decoder(x)


class OutputBlock(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes: int = 1):
        super().__init__()
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ch_in, int(ch_in / 4), 2, 2, 0),
            nn.BatchNorm2d(int(ch_in / 4)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(int(ch_in / 4), ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                ch_out,
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.output(x)


class NeoUnet(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        hardnet68 = torch.hub.load(
            "PingoLH/Pytorch-HarDNet", "hardnet68", pretrained=True
        )  # nosec
        child = hardnet68.base
        self.block1 = nn.Sequential(*child[0:2])  # N x 64 x 56 x 56
        self.block2 = nn.Sequential(*child[2:5])  # N x 128 x 56 x 56
        self.block3 = nn.Sequential(*child[5:10])  # N x 320 x 28 x 28
        self.block4 = nn.Sequential(*child[10:13])  # N x 640 x 14 x 14
        self.block5 = nn.Sequential(*child[13:16])  # N x 1024 x 7 x 7

        self.ag6 = SCGate(F_g=1024, F_l=640, size=2)
        self.up6 = Up2Convolution3(ch_in=1024, ch_out=640)
        self.decoder6 = DecoderBlock(ch_in=1280, ch_out=640)
        self.out6 = nn.Upsample(size=128, mode="bilinear")

        self.ag7 = SCGate(F_g=640, F_l=320, size=2)
        self.up7 = Up2Convolution3(ch_in=640, ch_out=320)
        self.decoder7 = DecoderBlock(ch_in=640, ch_out=320)
        self.out7 = nn.Upsample(size=128, mode="bilinear")

        self.ag8 = SCGate(F_g=320, F_l=128, size=2)
        self.up8 = Up2Convolution3(ch_in=320, ch_out=128)
        self.decoder8 = DecoderBlock(ch_in=256, ch_out=128)
        self.out8 = nn.Upsample(size=128, mode="bilinear")

        self.ag9 = SCGate(F_g=128, F_l=64, size=2)
        self.up9 = Up2Convolution3(ch_in=128, ch_out=64)
        self.decoder9 = DecoderBlock(ch_in=128, ch_out=64)
        self.out9 = OutputBlock(ch_in=1152, ch_out=64, n_classes=n_classes)

    def forward(self, X):
        """ "
        @param: X(tensor) - N x C x H x W
        @return: out6, out7, out8, out9 - (14, 28, 56, 112)
        """
        # Encoder
        encoder1 = self.block1(X)
        encoder2 = self.block2(encoder1)
        encoder3 = self.block3(encoder2)
        encoder4 = self.block4(encoder3)

        # Center
        center = self.block5(encoder4)

        # Decoder
        att_gate6 = self.ag6(center, encoder4)
        up2_conv6 = self.up6(center)
        cat6 = torch.cat((att_gate6, up2_conv6), dim=1)
        decoder6 = self.decoder6(cat6)
        output6 = self.out6(decoder6)

        att_gate7 = self.ag7(decoder6, encoder3)
        up2_conv7 = self.up7(decoder6)
        cat7 = torch.cat((att_gate7, up2_conv7), dim=1)
        decoder7 = self.decoder7(cat7)
        output7 = self.out7(decoder7)

        att_gate8 = self.ag8(decoder7, encoder2)
        up2_conv8 = self.up8(decoder7)
        cat8 = torch.cat((att_gate8, up2_conv8), dim=1)
        decoder8 = self.decoder8(cat8)
        output8 = self.out8(decoder8)

        att_gate9 = self.ag9(decoder8, encoder1)
        up2_conv9 = self.up9(decoder8)
        cat9 = torch.cat((att_gate9, up2_conv9), dim=1)
        decoder9 = self.decoder9(cat9)
        cat_all = torch.cat((decoder9, output8, output7, output6), dim=1)
        output9 = self.out9(cat_all)

        return output9


class OutputBlock1(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes: int = 1):
        super().__init__()
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ch_in, int(ch_in / 4), 2, 2, 0),
            nn.BatchNorm2d(int(ch_in / 4)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(int(ch_in / 4), ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                ch_out,
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.output(x)


class NeoUnet1(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        hardnet68 = torch.hub.load(
            "PingoLH/Pytorch-HarDNet", "hardnet68", pretrained=True
        )  # nosec
        child = hardnet68.base
        self.block1 = nn.Sequential(*child[0:2])  # N x 64 x 56 x 56
        self.block2 = nn.Sequential(*child[2:5])  # N x 128 x 56 x 56
        self.block3 = nn.Sequential(*child[5:10])  # N x 320 x 28 x 28
        self.block4 = nn.Sequential(*child[10:13])  # N x 640 x 14 x 14
        self.block5 = nn.Sequential(*child[13:16])  # N x 1024 x 7 x 7

        self.ag6 = SCGate(F_g=1024, F_l=640, size=2)
        self.up6 = Up2Convolution3(ch_in=1024, ch_out=640)
        self.decoder6 = DecoderBlock(ch_in=1280, ch_out=640)
        self.out6 = nn.Upsample(size=128, mode="bilinear")

        self.ag7 = SCGate(F_g=1024, F_l=320, size=4)
        self.up7 = Up2Convolution3(ch_in=640, ch_out=320)
        self.decoder7 = DecoderBlock(ch_in=640, ch_out=320)
        self.out7 = nn.Upsample(size=128, mode="bilinear")

        self.ag8 = SCGate(F_g=1024, F_l=128, size=8)
        self.up8 = Up2Convolution3(ch_in=320, ch_out=128)
        self.decoder8 = DecoderBlock(ch_in=256, ch_out=128)
        self.out8 = nn.Upsample(size=128, mode="bilinear")

        self.ag9 = SCGate(F_g=1024, F_l=64, size=16)
        self.up9 = Up2Convolution3(ch_in=128, ch_out=64)
        self.decoder9 = DecoderBlock(ch_in=128, ch_out=64)
        self.out9 = OutputBlock1(ch_in=1152, ch_out=64, n_classes=n_classes)

    def forward(self, X):
        """ "
        @param: X(tensor) - N x C x H x W
        @return: out6, out7, out8, out9 - (14, 28, 56, 112)
        """
        # Encoder
        encoder1 = self.block1(X)
        encoder2 = self.block2(encoder1)
        encoder3 = self.block3(encoder2)
        encoder4 = self.block4(encoder3)

        # Center
        center = self.block5(encoder4)

        # Decoder
        att_gate6 = self.ag6(center, encoder4)
        up2_conv6 = self.up6(center)
        cat6 = torch.cat((att_gate6, up2_conv6), dim=1)
        decoder6 = self.decoder6(cat6)
        output6 = self.out6(decoder6)

        att_gate7 = self.ag7(center, encoder3)
        up2_conv7 = self.up7(decoder6)
        cat7 = torch.cat((att_gate7, up2_conv7), dim=1)
        decoder7 = self.decoder7(cat7)
        output7 = self.out7(decoder7)

        att_gate8 = self.ag8(center, encoder2)
        up2_conv8 = self.up8(decoder7)
        cat8 = torch.cat((att_gate8, up2_conv8), dim=1)
        decoder8 = self.decoder8(cat8)
        output8 = self.out8(decoder8)

        att_gate9 = self.ag9(center, encoder1)
        up2_conv9 = self.up9(decoder8)
        cat9 = torch.cat((att_gate9, up2_conv9), dim=1)
        decoder9 = self.decoder9(cat9)
        cat_all = torch.cat((decoder9, output8, output7, output6), dim=1)
        output9 = self.out9(cat_all)

        return output9


if __name__ == "__main__":
    x = torch.randn((4, 3, 128, 128))
    model = NeoUnet()
    y = model(x)
    print(y.shape)
