# git will not automatically track this file
# git will not automatically track this file
from uses_hpc import USES_HPC

WANDB_RECORDING = True
SAVE_MODEL = True
SCHEDULAR = True
LOWERCASE = True
PAIRS = True
BIGRAM = True #Bigram is incmpatable with Pairs
if BIGRAM:
    PAIRS = False
CHARACTER_SEPERATION = True
LSTM_DROPOUT = True