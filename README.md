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

Trained attack models, used Datasets (cited in the paper), and extracted embeddings are publicly available in [Zenodo repository](https://zenodo.org/records/15639971). Please create directories accordingly for reproduction.


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
@inproceedings{chen-etal-2025-algen,
    title = "{ALGEN}: Few-shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment and Generation",
    author = "Chen, Yiyi  and
      Xu, Qiongkai  and
      Bjerva, Johannes",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1185/",
    doi = "10.18653/v1/2025.acl-long.1185",
    pages = "24330--24348",
    ISBN = "979-8-89176-251-0"
}
```

### Disclaimer

The open-sourced code, vectors and models are for research purpose only.


