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


def char_confidence(output):

    probs = torch.softmax(output, dim=2)
    max_probs, _ = probs.max(dim=2)

    return max_probs

# =========================
# CTC DECODING
# =========================
def ctc_greedy_decode(output, blank=0):

    probs = torch.softmax(output, dim=2)
    max_probs, preds = torch.max(probs, dim=2)

    decoded = []
    confidences = []

    for b in range(preds.size(0)):

        prev = blank
        string = []
        char_conf = []

        for t in range(preds.size(1)):

            p = preds[b, t].item()

            if p != blank and p != prev:
                string.append(idx_to_char.get(p, ""))
                char_conf.append(max_probs[b, t].item())

            prev = p

        decoded.append("".join(string))
        confidences.append(torch.tensor(char_conf))
    return decoded, confidences

def compute_cer(predictions, ground_truths):

    total_distance = 0
    total_chars = 0
    
    for pred, gt in zip(predictions, ground_truths):
        total_distance += Levenshtein.distance(pred, gt)
        total_chars += len(gt)
    
    return total_distance / total_chars if total_chars > 0 else 0

def ctc_bigram_decode(output, bigram_lm, blank=0, lm_weight=0.3):
    """
    CTC decoding with a simple bigram LM, returning both decoded text and per-character confidence.
    """

    probs = torch.softmax(output, dim=2)
    log_probs = output.log_softmax(2)
    preds = torch.argmax(log_probs, dim=2)

    decoded = []
    confidences = []

    for b, pred in enumerate(preds):
        prev = "<s>"
        prev_ctc = blank
        string = []
        char_conf = []

        for t, p in enumerate(pred):
            p = p.item()

            if p == blank or p == prev_ctc:
                prev_ctc = p
                continue

            char = idx_to_char.get(p, "")

            # LM bonus (optional, does not affect confidence)
            lm_bonus = 0
            if prev in bigram_lm and char in bigram_lm[prev]:
                lm_bonus = bigram_lm[prev][char]

            score = log_probs[b, t, p].item() + lm_weight * lm_bonus

            # append decoded char and its confidence
            string.append(char)
            char_conf.append(probs[b, t, p].item())

            prev = char
            prev_ctc = p

        decoded.append("".join(string))
        confidences.append(torch.tensor(char_conf))

    return decoded, confidences

def ctc_beam_decode(outputs,decoder):

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(outputs)

    decoded = []

    for i in range(outputs.size(0)):
        length = out_lens[i][0]
        tokens = beam_results[i][0][:length]

        text = "".join([idx_to_char[t.item()] for t in tokens if t.item() != blank_label])

        decoded.append(text)

    return decoded

def ctc_beam_bigram_decode(outputs, bigram_lm, decoder, beam_width=10, lm_weight=0.3):
    """
    Beam decode a batch of CTC outputs using a bigram LM.
    """
    decoded = []
    log_probs = outputs.log_softmax(2).cpu().numpy()

    for lp in log_probs:
        beams = decoder.decode_beams(lp, beam_width=beam_width)
        best_text = None
        best_score = -float("inf")

        for beam in beams:
            text = beam[0]  # decoded text
            logit_score = beam[3]  # <--- use 4th element
            # Compute LM score
            lm_score = bigram_score(text, bigram_lm)
            total_score = logit_score + lm_weight * lm_score

            if total_score > best_score:
                best_score = total_score
                best_text = text

        decoded.append(best_text)

    return decoded