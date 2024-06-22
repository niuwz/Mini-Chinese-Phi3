from dataclasses import dataclass
from os.path import dirname, abspath
import platform
from typing import List, Tuple
# replace '\' on windows to '/'
PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')
                             ) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))
DATA_ROOT: str = PROJECT_ROOT + "/dataset/"
TEMP_ROOT: str = PROJECT_ROOT + "/runtime/"


# ===================================================================================
# 以下为训练的配置

@dataclass
class MiniPhi3PreTrainConfig:
    epochs: int = 1
    per_device_batch_size: int = 40

    learn_rate: float = 1e-4

    gradient_accumulation_steps: int = 8

    # 模型参数预热步数
    warmup_steps: int = 1024

    model_config_file: str = PROJECT_ROOT + '/pretrain/model_config.json'
    train_files: Tuple[str] = (DATA_ROOT + "2020-40_zh_head_0000.parquet",
                               DATA_ROOT + "2020-40_zh_head_0001.parquet",
                               DATA_ROOT + "2020-40_zh_head_0002.parquet",
                               DATA_ROOT + "2020-40_zh_head_0003.parquet",
                               DATA_ROOT + "2020-40_zh_head_0004.parquet",
                               DATA_ROOT + "2020-40_zh_head_0005.parquet",
                               DATA_ROOT + "2020-40_zh_head_0006.parquet",
                               DATA_ROOT + "2020-40_zh_head_0007.parquet",
                               DATA_ROOT + "2020-40_zh_head_0008.parquet",
                               )
    eval_files: Tuple[str] = (DATA_ROOT + "2020-40_zh_head_0009.parquet",
                              )

    output_dir: str = TEMP_ROOT + 'miniphi3'
    tokenizer_dir: str = PROJECT_ROOT + '/pretrain/'
    model_file: str = PROJECT_ROOT + '/pretrain/'

    # Windows 使用默认的attention实现，
    attn_implementation: str = (
        "eager" if platform.system() == "Windows" else "flash_attention_2"
    )

    logging_steps: int = 10
    save_steps: int = 100

    keep_latest_n_ckp: int = 5

    max_seq_len: int = 512

# 以下为sft配置


@dataclass
class MiniPhi3SFTConfig:
    max_seq_len: int = 512
    pretrained_model: str = PROJECT_ROOT + '/pretrain/'
    output_dir: str = TEMP_ROOT + '/fine_tuned/sft'
    sft_data_file: str = PROJECT_ROOT + '/sft_train'
    per_device_train_batch_size: int = 40
    num_train_epochs: int = 2
    save_steps: int = 50
    gradient_accumulation_steps: int = 16
    full_ft_rate: int= 5
    gradient_checkpointing = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.02
    keep_latest_n_ckp: int = 5
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    logging_first_step: bool = True
    logging_steps: int = 10
    sft_model_file: str = PROJECT_ROOT + '/fine_tuned/sft'
    warmup_ratio: float = 0.03
    fp16 = False
    bf16 = True
    group_by_length = True

    sft_datasets: Tuple[str] = (DATA_ROOT + '/sft_data/belle_3M_cn.parquet',)
    # Windows 使用默认的attention实现，
    attn_implementation: str = (
        "eager" if platform.system() == "Windows" else "flash_attention_2"
    )

    # LoRA attention dimension
    lora_r: int = 64

    # Alpha parameter for LoRA scaling
    lora_alpha: int = 32

    # Dropout probability for LoRA layers
    lora_dropout: int = 0.1

    # Activate 4-bit precision base model loading
    use_4bit: bool = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype: str = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type: str = "fp4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant: bool = False


# 以下为dpo训练配置
@dataclass
class MiniPhi3DPOConfig:
    max_seq_len: int = 512
    sft_model_file: str = PROJECT_ROOT + '/fine_tuned/sft'
    output_dir: str = TEMP_ROOT + '/fine_tuned/dpo/'
    # Windows 使用默认的attention实现，
    attn_implementation: str = (
        "eager" if platform.system() == "Windows" else "flash_attention_2"
    )

    dpo_data_file: Tuple[str] = (DATA_ROOT + 'train-dpo.parquet', 
                                 DATA_ROOT + 'test-dpo.parquet',
                                 DATA_ROOT + 'rejected_dpo_alpaca_gpt4_data_zh.parquet')
    log_dir: str = PROJECT_ROOT + '/logs/'
    optim: str = "adamw_torch"
    weight_decay: float = 0.02
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 16
    logging_first_step: bool = True
    logging_steps: int = 20
    save_steps: int = 100
    eval_steps: int = 100
    dpo_model_file: str = PROJECT_ROOT + '/fine_tuned/dpo'
    warmup_ratio: float = 0.03
    beta: float = 0.1
    keep_latest_n_ckp:int = 5


    # LoRA attention dimension
    lora_r: int = 64

    # Alpha parameter for LoRA scaling
    lora_alpha: int = 32

    # Dropout probability for LoRA layers
    lora_dropout: int = 0.1


# ===================================================================================
# 以下为推断的配置
@dataclass
class InferConfig:
    max_seq_len: int = 512
    # 混合精度 ''no','fp16','bf16' or 'fp8'
    mixed_precision: str = "bf16"
    dpo_path:str = PROJECT_ROOT + "/fine_tuned/dpo/"
    sft_path:str = PROJECT_ROOT + "/fine_tuned/sft/"
    pretrained_path:str = PROJECT_ROOT + "/pretrain/"



