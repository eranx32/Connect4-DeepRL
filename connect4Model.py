import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.flatten_size = 128 * 5 * 6

        self.fc_common = nn.Linear(self.flatten_size, 512)

        self.fc_value = nn.Linear(512, 1)

        self.fc_advantage = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc_common(x))

        val = self.fc_value(x)
        adv = self.fc_advantage(x)


        return val + (adv - adv.mean(dim=1, keepdim=True))
