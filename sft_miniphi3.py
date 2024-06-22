import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import numpy as np
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from model.configuation_miniPhi3 import MiniPhiConfig
from model.modeling_miniphi3 import MiniPhi3
import pandas as pd
import time
from config import PROJECT_ROOT, DATA_ROOT, TEMP_ROOT, MiniPhi3SFTConfig


def format_example(example):
    prompt = "[user]\n{instruction}[end]\n[assistant]"
    target = example["output"] + "[EOS]"
    context = prompt.format_map(
        dict(instruction=example["instruction"]))
    example['text'] = context+target
    return example

def Full_Parameter_Fine_Tuning(sft_args, model, sft_dataset,tokenizer,sft_config):
    print("#"*30+"Full Parameter Fine Tuning"+"#"*30)
    sft_dataset = sft_dataset.map(
        function=format_example,
        remove_columns=sft_dataset.column_names)
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        tokenizer=tokenizer,
        args=sft_args,
    )
    # Train model
    trainer.train()
    # Save trained model
    print("##############\n",trainer.model)
    trainer.save_model(sft_config.sft_model_file)
    tokenizer.save_pretrained(sft_config.sft_model_file)
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/sft_full_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    del trainer

def LoRA_SFT(sft_args,sft_dataset,eval_dataset,peft_config,sft_config):
    print("#"*30+"Fine Tuning Use LoRA"+"#"*30)
    sft_dataset = sft_dataset.map(
        function=format_example,
        remove_columns=sft_dataset.column_names)
    eval_dataset = eval_dataset.map(
        function=format_example,
        remove_columns=eval_dataset.column_names)
    # miniphi_config = MiniPhiConfig(
    #     vocab_size=32064,
    #     attn_implementation=sft_config.attn_implementation,
    #     use_cope=False)

    miniphi_config = MiniPhiConfig.from_pretrained(
        sft_config.sft_model_file,
        attn_implementation=sft_config.attn_implementation,
        )
    # 根据全参数微调的模型继续进行PEFT
    model = MiniPhi3.from_pretrained(
        sft_config.sft_model_file,
        config=miniphi_config,
        torch_dtype=torch.bfloat16
    )
    # 在PEFT模型的基础上继续PEFT
    # base_model = MiniPhi3.from_pretrained(sft_config.sft_model_file, config=miniphi_config)
    # model = PeftModel.from_pretrained(base_model, sft_config.sft_model_file)
    # print(model)
    # model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(sft_config.sft_model_file)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_args,
    )
    # Train model
    trainer.train()
    # Save trained model
    trainer.save_model(sft_config.sft_model_file)
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/sft_lora_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    print(trainer.model)


if __name__ == "__main__":
    sft_config = MiniPhi3SFTConfig()
    miniphi_config = MiniPhiConfig(
        attn_implementation=sft_config.attn_implementation,
        use_cope=False)
    model = MiniPhi3.from_pretrained(
        sft_config.pretrained_model,
        config=miniphi_config,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_config.pretrained_model)
    tokenizer.padding_side = "left"
    model.resize_token_embeddings((len(tokenizer)//64+1)*64)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ['[user]', '[end]', '[assistant]']})


    # Set training parameters
    full_sft_dataset = load_dataset(
        path="parquet",
        data_files=sft_config.sft_datasets,
        split="train[:{}%]".format(sft_config.full_ft_rate),
        keep_in_memory=False,
        cache_dir=TEMP_ROOT + "data/.cache"
    )

    full_sft_args = SFTConfig(
        output_dir=sft_config.output_dir+"/full/",
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        num_train_epochs=sft_config.num_train_epochs,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        optim=sft_config.optim,
        save_steps=sft_config.save_steps,
        logging_steps=sft_config.logging_steps,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        fp16=sft_config.fp16,
        bf16=sft_config.bf16,
        # max_steps=1,
        warmup_ratio=0.01,
        group_by_length=sft_config.group_by_length,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        report_to="tensorboard",
        log_level="info",
        logging_first_step=True,
        dataset_text_field="text",
        max_seq_length=sft_config.max_seq_len,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        save_total_limit=sft_config.keep_latest_n_ckp,
        deepspeed="deepspeed_configs/ds_config.json"
    )


    # Set training parameters
    peft_sft_dataset = load_dataset(
        path="parquet",
        data_files=sft_config.sft_datasets,
        split="train[{}%:90%]".format(sft_config.full_ft_rate),
        keep_in_memory=False,
        cache_dir=TEMP_ROOT + "data/.cache"
    )

    eval_sft_dataset = load_dataset(
        path="parquet",
        data_files=sft_config.sft_datasets,
        split="train[90%:]",
        keep_in_memory=False,
        cache_dir=TEMP_ROOT + "data/.cache"
    )

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=sft_config.lora_alpha,
        lora_dropout=sft_config.lora_dropout,
        r=sft_config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["qkv_proj"]
        target_modules="all-linear",
    )
    print(model)

    lora_sft_args = SFTConfig(
        output_dir=sft_config.output_dir+"/peft/",
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        num_train_epochs=sft_config.num_train_epochs,
        per_device_eval_batch_size=sft_config.per_device_train_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        optim=sft_config.optim,
        eval_strategy="steps",
        eval_steps=sft_config.save_steps,
        save_steps=sft_config.save_steps,
        logging_steps=sft_config.logging_steps,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        fp16=sft_config.fp16,
        bf16=sft_config.bf16,
        # max_steps=1,
        warmup_ratio=sft_config.warmup_ratio,
        group_by_length=sft_config.group_by_length,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        report_to="tensorboard",
        log_level="info",
        logging_first_step=True,
        dataset_text_field="text",
        max_seq_length=sft_config.max_seq_len,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        save_total_limit=sft_config.keep_latest_n_ckp,
        deepspeed="deepspeed_configs/ds_config.json"
    )
    
    Full_Parameter_Fine_Tuning(full_sft_args,model,full_sft_dataset,tokenizer,sft_config)
    del model
    del full_sft_dataset
    del tokenizer
    torch.cuda.empty_cache()
    print(peft_sft_dataset)
    print(eval_sft_dataset)
    LoRA_SFT(lora_sft_args,peft_sft_dataset,eval_sft_dataset,peft_config,sft_config)
