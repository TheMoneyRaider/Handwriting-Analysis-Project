# git will not automatically track this file
WANDB_RECORDING = True
SAVE_MODEL = True
USES_HPC = False
SCHEDULAR = True
LOWERCASE = True
PAIRS = True
BIGRAM = True #Bigram is incmpatable with Pairs
if BIGRAM:
    PAIRS = False
CHARACTER_SEPERATION = True
LSTM_DROPOUT = True