# git will not automatically track this file
# git will not automatically track this file
from uses_hpc import USES_HPC

WANDB_RECORDING = True
SAVE_MODEL = True
SCHEDULAR = True
LOWERCASE = True
PAIRS = False
BIGRAM = True #Bigram is incmpatable with Pairs
CHARACTER_SEPERATION = False
LSTM_DROPOUT = True
BEAM_SEARCH = True
CONV_LAYER = True

if BIGRAM:
    PAIRS = False

#Tests TODO
#Beam no bigram
#dropout sweep
#actual wandb sweep
