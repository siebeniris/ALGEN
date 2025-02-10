#!/bin/bash -e

# WET

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-t5/t5-base 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google/mt5-base 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-bert/bert-base-multilingual-cased 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 sentence-transformers/gtr-t5-base 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 WET 0 0 0



# ========
sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-t5/t5-base 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google/mt5-base 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 sentence-transformers/gtr-t5-base 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 WET 0 0 0


# ========

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-t5/t5-base 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google/mt5-base 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 WET 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 sentence-transformers/gtr-t5-base 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 WET 0 0 0


# Shuffling


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-t5/t5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google/mt5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-bert/bert-base-multilingual-cased 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 sentence-transformers/gtr-t5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 Shuffling 0 0 0

# ========
sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-t5/t5-base 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google/mt5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 sentence-transformers/gtr-t5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 Shuffling 0 0 0

# ========

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-t5/t5-base 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google/mt5-base 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 Shuffling 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 sentence-transformers/gtr-t5-base 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 random 128 Shuffling 0 0 0



# NoDefense


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-t5/t5-base 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google/mt5-base 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 google-bert/bert-base-multilingual-cased 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 sentence-transformers/gtr-t5-base 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 text-embedding-ada-002 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 random 128 NoDefense 0 0 0


# ========
sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-t5/t5-base 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google/mt5-base 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 sentence-transformers/gtr-t5-base 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 text-embedding-ada-002 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 random 128 NoDefense 0 0 0

# ========

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-t5/t5-base 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google/mt5-base 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 google-bert/bert-base-multilingual-cased 128 NoDefense 0 0 0


sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 sentence-transformers/gtr-t5-base 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 text-embedding-ada-002 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 random 128 NoDefense 0 0 0
