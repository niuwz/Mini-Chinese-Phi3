
from logger import Logger
from model import modeling_miniphi3, configuation_miniPhi3
from datasets import Dataset, load_dataset
from config import PROJECT_ROOT, MiniPhi3PreTrainConfig, DATA_ROOT, TEMP_ROOT
from transformers.trainer_callback import TrainerControl, TrainerState
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer
)
torch.set_default_dtype(torch.bfloat16)
log = Logger('pretraun', save2file=True,
             file_name=PROJECT_ROOT + '/logs/pretrain.log')


class miniPhi3TrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        """
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


def token_to_id(tokenizer, max_seq_len):
    map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32

    def get_id(samples: dict) -> dict:
        batch_txt = samples["text"]
        outputs = tokenizer(
            batch_txt,
            padding="max_length",
            return_attention_mask=False,
            truncation=True,
            max_length=max_seq_len
        )
        input_ids = [np.array(item, dtype=map_dtype)
                     for item in outputs["input_ids"]]
        return {"input_ids": input_ids}
    return get_id


def get_maped_dataset(files, tokenizer, max_seq_len) -> Dataset:
    dataset = load_dataset(
        path="parquet",
        data_files=files,
        split="train",
        cache_dir=TEMP_ROOT + "data/.cache",
        keep_in_memory=False,
    )
    maped_dataset = dataset.map(
        token_to_id(tokenizer, max_seq_len),
        batched=True,
        batch_size=128,
        remove_columns=dataset.column_names,
        num_proc=24,
        keep_in_memory=False,
    )
    return maped_dataset


if __name__ == "__main__":
    pretrain_config = MiniPhi3PreTrainConfig()
    config = configuation_miniPhi3.MiniPhiConfig(
        attn_implementation=pretrain_config.attn_implementation,
        use_cope=False)
    model = modeling_miniphi3.MiniPhi3(config)
    print(model)
    tokenizer_path = PROJECT_ROOT+"/tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)

    trainer_callback = miniPhi3TrainerCallback()

    # `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataset = get_maped_dataset(
        pretrain_config.train_files, tokenizer, pretrain_config.max_seq_len)
    eval_dataset = get_maped_dataset(
        pretrain_config.eval_files, tokenizer, pretrain_config.max_seq_len)

    train_args = TrainingArguments(
        output_dir=pretrain_config.output_dir,
        per_device_train_batch_size=pretrain_config.per_device_batch_size,
        per_device_eval_batch_size=pretrain_config.per_device_batch_size,
        gradient_accumulation_steps=pretrain_config.gradient_accumulation_steps,
        num_train_epochs=pretrain_config.epochs,
        weight_decay=0.1,
        ddp_find_unused_parameters=False,
        warmup_steps=256,
        learning_rate=pretrain_config.learn_rate,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_strategy="steps",
        save_total_limit=pretrain_config.keep_latest_n_ckp,
        report_to="tensorboard",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        log_level="info",
        logging_first_step=True,
        deepspeed="deepspeed_configs/ds_config.json"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[trainer_callback],
    )
    trainer.train(resume_from_checkpoint=True)
    eval_results = trainer.evaluate()

    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model(pretrain_config.model_file)

    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
