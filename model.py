# Package imports
import torch
import torch.nn as nn
import editdistance

# File Imports
from dataset import *

# CNN Feature Extractor

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1))
        )

    def forward(self, x):
        return self.cnn(x)  # [B, C, H, W]


# =========================
# CRNN MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        
        self.rnn = nn.LSTM(
            input_size=256*8,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )


        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        features = self.cnn(x)
        b, c, h, w = features.size()
        
        output = features.permute(0,3,1,2).contiguous().view(b,w,c*h)

        output, _ = self.rnn(output)

        output = self.fc(output)

        return output

# =========================
# CTC DECODING
# =========================
def ctc_greedy_decode(output, blank=0):
    preds = torch.argmax(output, dim=2)

    decoded = []
    for pred in preds:
        prev = blank
        string = []
        for p in pred:
            p = p.item()
            if p != blank and p != prev:
                string.append(idx_to_char.get(p, ""))
            prev = p
        decoded.append("".join(string))
    return decoded

def compute_cer(predictions, ground_truths):

    total_distance = 0
    total_chars = 0
    
    for pred, gt in zip(predictions, ground_truths):
        total_distance += editdistance.eval(pred, gt)
        total_chars += len(gt)
    
    return total_distance / total_chars if total_chars > 0 else 0