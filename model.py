# Package imports
import torch
import torch.nn as nn
from rapidfuzz.distance import Levenshtein

# File Imports
from dataset import *
from meta_config import *

MODEL_FILENAME = "CNN_BiLSTM_W_CTC"

if BIGRAM:
    from bigram import *
if BEAM_SEARCH:
    from ctcdecode import CTCBeamDecoder
    decoder = CTCBeamDecoder(
        list(characters),
        beam_width=10,
        blank_id=blank_label
    )

# CNN Feature Extractor

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        if CHARACTER_SEPERATION:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,1)),
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1))
            )
        else:
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
        drop = 0.0
        input = 256*8
        if LSTM_DROPOUT:
            drop = 0.3
        if LSTM_INPUT_SIZE_MATCH:
            input = 512*8
        if CONV_LAYER:
            self.reduce = nn.Conv2d(512,256,1)
        self.rnn = nn.LSTM(
            input_size=input,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=drop
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
        total_distance += Levenshtein.distance(pred, gt)
        total_chars += len(gt)
    
    return total_distance / total_chars if total_chars > 0 else 0

def ctc_bigram_decode(output, bigram_lm, blank=0, lm_weight=0.3):

    log_probs = output.log_softmax(2)
    preds = torch.argmax(log_probs, dim=2)

    decoded = []

    for b, pred in enumerate(preds):

        prev = "<s>"
        prev_ctc = blank
        string = []

        for t, p in enumerate(pred):

            p = p.item()

            if p == blank or p == prev_ctc:
                prev_ctc = p
                continue

            char = idx_to_char.get(p, "")

            lm_bonus = 0
            if prev in bigram_lm and char in bigram_lm[prev]:
                lm_bonus = bigram_lm[prev][char]

            score = log_probs[b, t, p].item() + lm_weight * lm_bonus

            string.append(char)
            prev = char
            prev_ctc = p

        decoded.append("".join(string))

    return decoded

def ctc_beam_decode(outputs):

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(outputs)

    decoded = []

    for i in range(outputs.size(0)):
        length = out_lens[i][0]
        tokens = beam_results[i][0][:length]

        text = "".join([idx_to_char[t.item()] for t in tokens if t.item() != blank_label])

        decoded.append(text)

    return decoded


def ctc_beam_bigram_decode(outputs, bigram_lm, beam_width=10, blank=0, lm_weight=0.3):

    log_probs = outputs.log_softmax(2)
    batch_size, T, C = log_probs.shape

    decoded = []

    for b in range(batch_size):

        beam = [("", blank, 0.0)]  
        # (string, previous_token, score)

        for t in range(T):

            new_beam = []

            for seq, prev_token, score in beam:

                for c in range(C):

                    logp = log_probs[b, t, c].item()

                    if c == blank:
                        new_beam.append((seq, blank, score + logp))
                        continue

                    if c == prev_token:
                        new_seq = seq
                    else:
                        char = idx_to_char.get(c, "")
                        new_seq = seq + char

                    lm_bonus = lm_weight * bigram_score(new_seq, bigram_lm)

                    new_score = score + logp + lm_bonus

                    new_beam.append((new_seq, c, new_score))

            new_beam = sorted(new_beam, key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_width]

        best_seq = beam[0][0]
        decoded.append(best_seq)

    return decoded