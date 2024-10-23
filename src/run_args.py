import os
from dataclasses import dataclass, field
from typing import Optional


import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

@dataclass
class ModelArguments:

    encoder_decoder_path: str = field(
        default="google/mt5-base",
        metadata={
            "help":(
                "The name or path to encoder decoder model."
            )
        }
    )

    embedder_model_name: str = field(
        default="intfloat/multilingual-e5-base",
        metadata = {
            "help": "Model to get embeddings from"
        },
    )

    embedder_model_api: Optional[str] = field(
        default=None, metadata={"help": "API to get embeddings from"}
    )
    embedder_torch_dtype: str = field(
        default="float32",
        metadata={
            "help": "torch dtype of embedder",
            "choices": ["float32", "float16", "bfloat16"],
        },
    )
    embedding_transform_strategy: str = field(
        default="repeat",
        metadata={
            "help": "Strategy for transforming from sentence embedding into sequence-level input for encoder-decoder",
        },
    )
    cache_dir: Optional[str]=field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface"
        }
    )

    max_seq_length: int=field(
        default=32, metadata={"help": "Maximum sequence length for tokenizer"}
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataArguments:

    # dataset_name: Optional[str] = field(
    #
    # )

    max_eval_samples:int =field(
        default=500,
        metadata={
            'help': (
                "Samples for evaluation"
            )
        }
    )

    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to saves/<random hash>."
        },
    )
    steps_per_epoch: int = field(
        default=500_000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=50.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    per_device_train_batch_size: int = field(
        default=256, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=256, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )
    # torch_compile: bool = True # for torch 2

    ##################### Experimental Settings ####################
    experiment: str = field(
        default="inversion",
        metadata={
            "required": False,
            "help": "Which experiment to run (defines model, loss func, dataset...) ",
            "choices": [
                "inversion",  # our model: projects and feeds to encoder-decoder
                "inversion_from_logits",
                "inversion_from_logits_emb",
                "inversion_decoder_only",  # baseline: use single embedding as input to a decoder
                "inversion_bow",
                "inversion_na",
                "reranking",
                "corrector",
                "corrector_encoder",
            ],
        },
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )
    exp_group_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this sweep / series of experiments",
        },
    )

    # Need to *not* remove unused columns so we keep query_attention_mask, etc.
    # which huggingface doesn't think we need.
    remove_unused_columns: bool = False

    # Do evaluation and logging on certain num steps.
    # evaluation_strategy: str = "steps"
    eval_strategy: str = "steps"  # transformer v4.41.2
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 2  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=4000, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=4000,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=40000,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )
    mock_embedder: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, will delete the embedder and replace all embedder logits with"
                " zeros once training starts. You probably don't want to do this. But "
                " if you precomputed all the embeddings for train and val, this will"
                " work fine, except the embedding-based metrics (just cosine similarity"
                " I think) will be broken."
            )
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    apply_early_stopping_metric: str = field(
        default="",
        metadata={
            "help": (
                "Whether to apply early stopping or not."
            )
        }
    )

    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        # num_workers = torch.cuda.device_count()

        # this should be the cpu not gpu.
        num_workers = 7  # set lower number to avoid out-of-memory
        # do not use os.cpu_count() it will see all the gpus. 128 from one node.
        print(f"num_workers {num_workers}")
        #  Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
        #  main process.

        # useful for RUST application such as python
        # It is especially useful in environments where you need to limit the number of CPU cores used by certain
        # parts of your application to prevent resource contention.
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers

        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        # defaults from SentenceTransformers
        # lr 2e-5
        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"
        # for scheduler.
        self.lr_scheduler_type = "constant_with_warmup"

        self.load_best_model_at_end = True
        self.greater_is_better = False

        self.do_eval = False
        # self.ddp_backend = "gloo"
