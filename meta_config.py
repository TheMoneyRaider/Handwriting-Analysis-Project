# git will not automatically track this file
# git will not automatically track this file
from uses_hpc import USES_HPC

WANDB_RECORDING = True
SAVE_MODEL = True
SCHEDULAR = True
LOWERCASE = True
BIGRAM = True
LSTM_DROPOUT = True
CONFIDENCE = True

BEAM_SEARCH = False
CONV_LAYER = False
CHARACTER_SEPERATION = False
PAIRS = False

if BIGRAM:
    PAIRS = False

#Tests TODO
#Beam no bigram
#dropout sweep
#actual wandb sweep
