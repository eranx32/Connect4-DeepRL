import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Model(nn.Module):
    def __init__(self):
        super().__init__()
        # חלק ה-Convolution נשאר זהה (העיניים)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.flatten_size = 128 * 5 * 6

        # --- כאן מתחיל ה-Dueling Architecture ---

        # שכבה משותפת אחרי ה-Flatten
        self.fc_common = nn.Linear(self.flatten_size, 512)

        # ראש 1: Value Stream (מעריך כמה המצב טוב באופן כללי) - מוציא מספר בודד
        self.fc_value = nn.Linear(512, 1)

        # ראש 2: Advantage Stream (מעריך את היתרון של כל פעולה) - מוציא 7 מספרים
        self.fc_advantage = nn.Linear(512, 7)

    def forward(self, x):
        # Convolution Layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.flatten(start_dim=1)

        # Common FC Layer
        x = F.relu(self.fc_common(x))

        # Split to Value and Advantage
        val = self.fc_value(x)  # (Batch, 1)
        adv = self.fc_advantage(x)  # (Batch, 7)

        # איחוד מחדש לפי הנוסחה של Dueling DQN:
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        # זה טריק מתמטי שמייצב את הלמידה
        return val + (adv - adv.mean(dim=1, keepdim=True))
