#!/bin/bash -e

sbatch lumi_classifiers_gaussian_noise.sh google-t5/t5-base

sbatch lumi_classifiers_gaussian_noise.sh google/mt5-base


sbatch lumi_classifiers_gaussian_noise.sh sentence-transformers/gtr-t5-base

sbatch lumi_classifiers_gaussian_noise.sh google-bert/bert-base-multilingual-cased

sbatch lumi_classifiers_gaussian_noise.sh text-embedding-ada-002

