#!/bin/bash -e




sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 random 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 random 128 NoDefense 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 random 128 NoDefense 0 0 0



sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 random 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 random 128 WET 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 random 128 WET 0 0 0



sbatch lumi_classifiers.sh yiyic/snli_ds nli 3 random 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/sst2_ds sentiment 2 random 128 Shuffling 0 0 0

sbatch lumi_classifiers.sh yiyic/s140_ds sentiment 2 random 128 Shuffling 0 0 0

#sbatch lumi_classifiers_gaussian_noise.sh random
#
#
#sbatch lumi_classifiers_ldp_model.sh random