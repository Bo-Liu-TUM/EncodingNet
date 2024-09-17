import torch
import torch.nn as nn
# class BasicBlock(nn.Module):
#     def __init__(self, ):
#         super(BasicBlock, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(16)
#
#     def forward(self, x):
#         identity = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x += identity
#         out = self.relu(x)
#         return out
#
#
# class Decoder(nn.Module):
#     def __init__(self, bit=8):
#         super(Decoder, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.dec1_fc = nn.Linear(1, 16)
#         self.dec1_bn = nn.BatchNorm1d(16)
#         self.dec2_fc = nn.Linear(1, 16)
#         self.dec2_bn = nn.BatchNorm1d(16)
#
#     def forward(self, x1, x2):
#         x1 = self.dec1_fc(x1)
#         x1 = self.dec1_bn(x1)
#         x1 = self.relu(x1)
#         x2 = self.dec2_fc(x2)
#         x2 = self.dec2_bn(x2)
#         x2 = self.relu(x2)
#         out = (x1.unsqueeze(2) @ x2.unsqueeze(1)).unsqueeze(1)
#         return out
#
#
# class NN_LUT(nn.Module):
#
#     def __init__(self, bit: [int, None] = 8, product_bit: [int, None] = 42):
#         super(NN_LUT, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.decoder = Decoder(8)
#         self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.first_bn = nn.BatchNorm2d(16)
#         self.layer1 = BasicBlock()
#         self.layer2 = BasicBlock()
#         self.layer3 = BasicBlock()
#         self.last_fc = nn.Linear(4096, product_bit)
#
#     def forward(self, x1, x2):
#         x = self.decoder(x1, x2)
#         x = self.first_conv(x)
#         x = self.first_bn(x)
#         x = self.relu(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = torch.flatten(x, 1)
#         out = self.last_fc(x)
#         return out



class BasicBlock(nn.Module):
    def __init__(self, ):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x += identity
        out = self.relu(x)
        return out


class NN_LUT(nn.Module):

    def __init__(self, bit: [int, None] = 8, product_bit: [int, None] = 42):
        super(NN_LUT, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.first_fc = nn.Linear(2, 16)
        # self.first_fc_x1 = nn.Linear(1, 16)
        # self.first_fc_x2 = nn.Linear(1, 16)
        self.first_bn = nn.BatchNorm1d(16)
        self.second_fc = nn.Linear(16, 64)
        self.second_bn = nn.BatchNorm1d(64)
        self.layer1 = BasicBlock()
        self.layer2 = BasicBlock()
        self.layer3 = BasicBlock()
        self.last_fc = nn.Linear(64, product_bit)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=1)
        x = self.first_fc(x)
        # x1 = self.first_fc_x1(x1)
        # x2 = self.first_fc_x1(x2)
        # x = x1 + x2
        x = self.first_bn(x)
        x = self.relu(x)
        x = self.second_fc(x)
        x = self.second_bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.last_fc(x)
        return out

