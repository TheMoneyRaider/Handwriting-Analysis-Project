from collections import defaultdict
import math
from meta_config import *
import tempfile

def build_bigram_lm(dataset):

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for _, label in dataset:
        text = label.lower()

        prev = "<s>"
        for c in text:
            counts[prev][c] += 1
            totals[prev] += 1
            prev = c

    bigram_log_probs = {}

    for prev in counts:
        bigram_log_probs[prev] = {}
        for c in counts[prev]:
            prob = counts[prev][c] / totals[prev]
            bigram_log_probs[prev][c] = math.log(prob)

    return bigram_log_probs


def bigram_score(beam_text, bigram_lm, lm_weight=1.0):
    score = 0
    prev = "<s>"
    for c in beam_text:
        if prev in bigram_lm and c in bigram_lm[prev]:
            score += bigram_lm[prev][c]
        prev = c
    return lm_weight * score