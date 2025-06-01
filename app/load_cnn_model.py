import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio

TARGET_LEN = 64000
LABELS_MAP = {'en': 0, 'ja': 1, 'vi': 2}
IDX_TO_LANG = {v: k for k, v in LABELS_MAP.items()}

# ---------- MÔ HÌNH CNN ----------
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 5, stride=2)
        self.bn3 = nn.BatchNorm1d(64)

        with torch.no_grad():
            dummy = torch.randn(1, 1, TARGET_LEN)
            x = self.forward_features(dummy)
            self.flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, len(LABELS_MAP))

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

