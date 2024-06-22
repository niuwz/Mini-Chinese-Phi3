from data_process import delete_file
from config import PROJECT_ROOT, DATA_ROOT, TEMP_ROOT, InferConfig
# from logger import Logger
# from model.infer import ChatBot
import pyarrow.parquet as pq
from rich import progress
import ujson
import numpy as np
import pandas as pd
import torch
import re
import os
import sys
from fastparquet import ParquetFile, write
from os.path import dirname, abspath, exists, isdir
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

DATA_ROOT = "/hy-tmp/"


def format_example(example):
    prompt = "[user]\n{text}[end]\n[assistant]"
    context = example["instruction"]+example["input"]
    context = prompt.format_map(
        dict(text=context))
    chosen = example["output"]
    return {
        "prompt": context,
        "chosen": chosen
    }


def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    ''' 
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True

    write(file_name, data_frame, compression='GZIP', append=append)


def process_alpaca_gpt4_data(max_len: int = 512) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
    '''

    read_file = DATA_ROOT + '/alpaca_gpt4_data_zh.json'
    save_file = DATA_ROOT + '/dpo_data/alpaca_gpt4_data_zh.json'

    my_data = []

    with open(read_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        print('length of {} is {}'.format(read_file, len(data)))
        for item in progress.track(data):
            prompt = item['instruction']
            inputs = item['input']
            response = item['output']

            if len(response) > max_len:
                continue  # 超长的不要

            if len(inputs.strip()) > 0:
                prompt = f"{prompt}，{inputs}"

            if len(prompt) > max_len:
                continue

            if len(prompt) == 0 or len(response) == 0:
                continue

            my_data.append(
                {
                    'prompt': prompt,
                    'chosen': response
                }
            )

    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)


def generate_alpaca_gpt4_reject_response(groups_cnt: int = 50000, max_len: int = 320, batch_size: int = 32) -> None:
    '''生成不是很满意的回答回答
    '''
    finetune_file = DATA_ROOT + '/alpaca_gpt4_data_zh.json'
    save_file = DATA_ROOT + '/dpo_data/rejected_dpo_alpaca_gpt4_data_zh.parquet'
    delete_file(save_file)

    alp_data = load_dataset("json",
                            data_files=finetune_file,
                            split="train",
                            keep_in_memory=False)
    alp_data = alp_data.map(function=format_example,
                            batch_size=16, remove_columns=alp_data.column_names)

    # chat = MiniPhi3Infence(infence_arg, model_type)
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=2048)
    chat = LLM(model="fine_tuned/vllm")
    batch_size = 32

    model_outs = []
    data_prompt = []
    data_chose = []

    for i in range(0, len(alp_data), batch_size):
        # 模型生成的答案为拒绝答案
        batch = alp_data[i:i+batch_size]
        if i % 100 == 0:
            print('process {} batchs.'.format(i))
        with torch.no_grad():
            outputs = chat.generate(batch["prompt"], sampling_params)
            output = [o.outputs[0].text for o in outputs]
        data_prompt.extend(batch["prompt"])
        data_chose.extend(batch["chosen"])
        model_outs.extend(output)
        print(len(model_outs))

        if len(model_outs) >= 2000:
            write_out = []
            for j in range(len(model_outs)):
                write_out.append({"prompt": data_prompt[j],
                                  "chosen": data_chose[j],
                                  "rejected": model_outs[j]})
            model_outs = []
            df = pd.DataFrame(write_out)
            write_single_parquet_file(save_file, df)

    if len(model_outs) > 0:
        write_out = []
        for i in range(len(model_outs)):
            write_out.append({"prompt": data_prompt[i],
                              "chosen": data_chose[i],
                              "rejected": model_outs[i]})
        model_outs = []
        df = pd.DataFrame(write_out)
        write_single_parquet_file(save_file, df)
    rd = load_dataset("parquet", data_files=save_file)
    print(alp_data)
    print(rd)


def replace_line(s: str) -> str:
    '''将双斜杠替换为单斜杠，既是 \\n 替换为 \n
    '''
    return re.sub('\\\\n', '\n', s)


if __name__ == "__main__":
    generate_alpaca_gpt4_reject_response()
