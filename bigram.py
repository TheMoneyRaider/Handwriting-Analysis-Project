from collections import defaultdict
import math

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