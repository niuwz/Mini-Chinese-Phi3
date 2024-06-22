import os
from config import PROJECT_ROOT, DATA_ROOT, TEMP_ROOT, MiniPhi3DPOConfig
import time
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from model.configuation_miniPhi3 import MiniPhiConfig
from model.modeling_miniphi3 import MiniPhi3


def format_example(example):
    prompt = "[user]\n{text}[end]\n[assistant]"
    if "[user]" in example["prompt"]:
        context = example["prompt"]
    else:
        context = prompt.format_map(
            dict(text=example["prompt"]))
    chosen = example["chosen"] + "[EOS]"
    reject = example["rejected"] + "[EOS]"
    return {
        "prompt": context,
        "chosen": chosen,
        "rejected": reject
    }


if __name__ == "__main__":
    dpo_config = MiniPhi3DPOConfig()
    miniphi_config = MiniPhiConfig.from_pretrained(dpo_config.sft_model_file,
                    attn_implementation = dpo_config.attn_implementation,
    )
    model = MiniPhi3.from_pretrained(
        dpo_config.sft_model_file,
        config=miniphi_config,
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, dpo_config.sft_model_file)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(dpo_config.sft_model_file)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
    miniphi_config.save_pretrained(dpo_config.dpo_model_file)
    peft_config = LoraConfig(
        lora_alpha=dpo_config.lora_alpha,
        lora_dropout=dpo_config.lora_dropout,
        r=dpo_config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["qkv_proj"]
        target_modules="all-linear"
    )

    dpo_data = load_dataset('parquet', data_files=dpo_config.dpo_data_file,
                            split='train', keep_in_memory=False)
    dpo_data = dpo_data.map(function=format_example,remove_columns=dpo_data.column_names)
    dpo_data = dpo_data.train_test_split(test_size=0.2, seed=0)
    print("#"*30+"Train Dataset" + "#"*30)
    print(dpo_data)

    dpo_args = DPOConfig(
        output_dir=dpo_config.output_dir,
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_config.per_device_eval_batch_size,
        num_train_epochs=dpo_config.num_train_epochs,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        optim=dpo_config.optim,
        max_length=dpo_config.max_seq_len,
        max_prompt_length=dpo_config.max_seq_len//2,
        save_strategy="steps",
        save_steps=dpo_config.save_steps,
        learning_rate=dpo_config.learning_rate,
        weight_decay=dpo_config.weight_decay,
        bf16=True,
        # max_steps=1,
        warmup_ratio=dpo_config.warmup_ratio,
        report_to="tensorboard",
        log_level="info",
        logging_first_step=True,
        save_total_limit=dpo_config.keep_latest_n_ckp,
        remove_unused_columns=False,
        deepspeed="deepspeed_configs/ds_config.json"
    )
    trainer = DPOTrainer(
        model=model,
        train_dataset=dpo_data["train"],
        eval_dataset=dpo_data["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=dpo_args,
    )

    trainer.train()
    trainer.save_model(dpo_config.dpo_model_file)
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(
        f"./logs/dpo_lora_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    print(trainer.model)
