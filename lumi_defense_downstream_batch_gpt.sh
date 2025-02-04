#!/bin/bash -e


# NoDefense
sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 NoDefense 0 0 0



# WET

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 WET 0 0 0



# Shuffling


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 Shuffling 0 0 0

