# ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation

__Yiyi Chen__, Qiongkai Xu (Correspondence author), Johannes Bjerva


[//]: # (![ALGEN]&#40;Figure2_07.pdf&#41;)

## Setup Conda Environment
```
conda create -n fewshot python=3.12

pip3 install torch torchvision torchaudio

pip3 install -r requirements.txt
```

## Materials

Trained attack models, used Datasets (cited in the paper), and extracted embeddings are publicly available in [Zenodo repository](https://zenodo.org/records/15639971).


## Few-shot Inversion Attack
### 1. Train Embedding-to-Text Generator 

```
python src/exp.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}\
    --max_length ${MAX_LENGTH} \
    --data_folder ${DATA_FOLDER} \
    --lang ${LANG}\
    --train_samples ${TRAIN_SAMPLES} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_epochs ${NUM_EPOCHS} \
    --wandb_run_name ${WAND_RUN_NAME} \
    --training_mode ${TRAINING_MODE}

```


### 2. Embedding alignment and Inversion Attack 

* with the Checkpoint from 1

```
python src/attacker_gt.py ${CHECKPOINT_PATH}

```

## Defense Mechanisms

### Attack Protect Embeddings 

* with the Checkpont from 1

```
python src/attacker_defended_embeds.py ${CHECKPOINT_PATH}
```

### Utility Test 

```
python -m src.classifiers.trainer ${dataset_name} ${task_name} ${num_labels} ${model_name} ${batch_size} ${defense_method} ${noise_level} ${delta} ${epsilon}

```


### Citation 
```
@misc{chen2025algenfewshotinversionattacks,
      title={ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation}, 
      author={Yiyi Chen and Qiongkai Xu and Johannes Bjerva},
      year={2025},
      eprint={2502.11308},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.11308}, 
}
```


