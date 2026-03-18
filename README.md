# Handwriting Recognition — AI 535 Final Project

A CRNN-BiLSTM model for offline handwriting recognition, trained on the IAM line-level dataset. This was our final project for AI 535 (Deep Learning) at [University Name].

---

## What This Is

Handwriting recognition is a well-explored problem. CNNs are good at pulling visual features out of images, but handwriting isn't just a collection of characters — it's a sequence. Letters look different depending on what's next to them, and spacing is inconsistent. Early CNN-only approaches struggled with this.

The field settled on a pretty standard stack: a CNN for feature extraction, a BiLSTM for sequence modeling, and CTC loss to handle the alignment between image width and variable-length output. That's what we built on.

---

## Our Approach

Rather than just reimplementing a known architecture, we split the project into two tracks:

**Vidyarthi** read the relevant literature and built a baseline CRNN-BiLSTM model derived from [[2]](#references). This gave us a documented starting point with a best CER of **7.09%**.

**Stober** then took that model and iterated on it using only knowledge from the course — no reading dominant papers, no looking up what others had tried. The goal was to see what independent reasoning could find in a well-explored space.

We focused on line-level recognition using the IAM dataset (~13,000 images), which has enough variety in handwriting style, source, and content that models can't rely on context to cheat their way to good performance.

---

## What We Tried

### ✅ Things That Helped

**Fixed-Rate Scheduler** — A plateau scheduler didn't move the needle (7.58% CER), but a fixed-rate epoch-based scheduler produced consistent minor improvements and was kept going forward.

**Case-Insensitive Output** — A lot of our error came from capitalization mismatches. Forcing lowercase output simplified the problem, and the network actually learned to map uppercase inputs to lowercase outputs on its own. Letters with similar shapes across cases (L/l, M/m, etc.) helped more than letters with different shapes (G/g, R/r) hurt.

**Bigram Language Model** — Built a two-character probability dictionary from the ground truths and used it to nudge outputs toward more probable sequences. Produced consistent across-the-board improvement, though it didn't beat the baseline minimum on its own.

**Dropout** — The biggest win. The baseline model didn't use dropout at all. Adding it at a rate of 0.3 dropped CER to 6.59%. Pushing to 0.4 got us to **6.55%**. Simple fix, meaningful result.

### ❌ Things That Didn't

**Pairwise Outputs** — The idea was that recognizing character pairs might help with ligatures and connected strokes. In practice it slowed learning significantly and raised CER across all epochs.

**Pooling Tweaks** — The first max pool layers were 1:1 ratio while later ones were 2:1. Seemed inconsistent for wide, short inputs. Changing it made things much worse — it compressed horizontal detail that the model needed to distinguish characters. Stopped after 5 epochs.

**Beam Search** — Had used it in a prior project, figured it might help CTC decoding. No notable improvement over greedy decoding.

**Confidence Weighing** — The idea was to combine the model's full confidence vector with bigram probabilities for better output. Ran out of time to fully implement it. Both versions tested didn't improve CER.

---

## Results

The best model combined case-insensitive output with 0.4 dropout. It beat the baseline at every single epoch and finished at **6.55% CER** vs the baseline's **7.09%**.

Both models made similar types of mistakes — characters that look nearly identical in handwriting (like `l` and `1`, or clusters of letters that blur together). The dropout model was noticeably better at resolving ambiguous clusters by the 15-epoch mark.

---

## What We'd Do Next

- Finish the confidence weighing implementation — the hypothesis is still sound, it just needed more time
- Keep pushing dropout rates to find the optimum
- Build the auxiliary line-splitting model to handle passage-level input, not just pre-segmented lines

---

## References

1. A. Baldominos, Y. Sáez, and P. Isasi, "Evolutionary convolutional neural networks: An application to handwriting recognition," *Neurocomputing*, vol. 283, pp. 38–52, 2018.
2. F. Kizilirmak and B. Yanikoglu, "CNN-BiLSTM model for English Handwriting Recognition: Comprehensive Evaluation on the IAM Dataset," arXiv:2307.00664, 2023.
3. M. Rabi and M. Amrouche, "Convolutional Arabic handwriting recognition system based on BLSTM-CTC using a Word Beam Search decoder," *Int. J. Intell. Syst. Appl. Eng.*, vol. 12, no. 16S, pp. 535–548, 2024.