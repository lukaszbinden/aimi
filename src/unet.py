# cf. https://www.youtube.com/watch?v=u1loyDCoGbE

import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def up_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.final_conv_1x1 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv1(image)  # skip connection
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)    # skip connection
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)    # skip connection
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)    # skip connection
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)
        print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        y = crop_tensor(x7, x)
        xx2 = self.up_conv_1(torch.cat([x, y], 1))

        xx3 = self.up_trans_2(xx2)
        y = crop_tensor(x5, xx3)
        xx4 = self.up_conv_2(torch.cat([xx3, y], 1))

        xx5 = self.up_trans_3(xx4)
        y = crop_tensor(x3, xx5)
        xx6 = self.up_conv_3(torch.cat([xx5, y], 1))

        xx7 = self.up_trans_4(xx6)
        y = crop_tensor(x1, xx7)
        xx8 = self.up_conv_4(torch.cat([xx7, y], 1))

        result = self.final_conv_1x1(xx8)

        print(result.size())
        # print(x.size())
        # print(x7.size())
        # print(y.size())
        # print(x.size())
        return result


if __name__ == '__main__':
    image = torch.rand(1, 1, 572, 572)
    model = UNet()
    print(model(image))
