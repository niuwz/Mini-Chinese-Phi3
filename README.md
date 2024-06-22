<div align="center">

# Mini Chinese Phi3

</div>


# 介绍

Mini-Chinese-Phi3是一个基于phi3模型结构的小型对话模型，总参数量约0.13B，使用常见的中文语料进行预训练和微调。主要内容包括了
- 数据集的整理与简单清洗
- 中文词表预训练
- 基于phi3结构的模型预训练
- 基于预训练模型的指令微调（SFT），包括了全量微调和LoRA微调
- 基于指令微调模型的直接偏好优化（DPO）
- 模型评测 **（待做）**

项目中的所有训练过程均在两张3090显卡上进行，使用DeepSpeed框架和Flash Attention 2进行加速，预训练用时约40小时，SFT和DPO微调共用时约8小时。本项目是我在学习LLM过程中的一个简单实践，同时也希望能够帮助到同样初学大模型的小伙伴。

模型参数现已开源，开放模型权重以供下载。项目地址：[Mini-Chinese-Phi3](https://huggingface.co/niwz/Mini-Chinese-Phi3)，可以通过`tokenizer = AutoTokenizer.from_pretrained("niwz/Mini-Chinese-Phi3")`和`model = AutoModelForCausalLM.from_pretrained("niwz/Mini-Chinese-Phi3")`直接加载模型参数。


# 目录结构

```
MiniChinesePhi3
├── datasets
│   ├── train_data_files
│   └── eval_data_files
├── pretrain
│   ├── pretrained_model_files
│   └── tokenizer_files
├── fine_tuned
│   ├── sft
|   |   ├── sft_model_files
|   |   └── tokenizer_files
│   └── dpo
|       ├── dpo_model_files
|       └── tokenizer_files
└── files_list_here
```

# 模型结构及训练过程

## 模型结构

本项目采用的参考模型结构是phi3，具体细节请参考原始论文：[Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone
](https://arxiv.org/abs/2404.14219)。
原计划将Phi3中的RoPE替换为最近发布的[CoPE](https://arxiv.org/abs/2405.18719)，但由于CoPE目前难以使用Flash Attention进行加速，因此Mini-Chinese-Phi3仍然使用了Phi3的RoPE。另一方面，参考GPT-2的参数规模，Mini-Chinese-Phi3的隐藏层维度为768，层数为12，词表数量为32000(SFT阶段调整为32064)，总参数量约0.13B。


## Tokenizer训练

   Wiki中文百科：[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) 

## 预训练

   天工数据集：https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main/data

注：受到硬件设备和精力限制，仅使用了天工数据集前10个文件进行预训练。

使用方式：
```bash
bash pretrain.sh
```
TODO: 补充预训练损失曲线

## SFT微调

SFT数据集使用了BelleGroup的指令数据集，包含约300万条中文指令数据。
- https://huggingface.co/datasets/BelleGroup/train_2M_CN
- https://huggingface.co/datasets/BelleGroup/train_1M_CN

在SFT阶段，由于聊天模板的缘故，引入了一些新的特殊token，为了使预训练模型能够适应这些特殊token，因此首先使用了5%的数据进行全量微调。

使用方式：
```bash
bash sft.sh
```

TODO: 补充SFT微调损失曲线

## DPO微调

本项目DPO数据集中的`chosen`文本来自alpaca数据集[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)，拒绝文本`rejected`来自SFT微调模型的输出，共计约5万条数据。生成DPO数据集时建议使用vLLM进行推理加速，否则生成速度会很慢。

使用方式：
```bash
bash dpo.sh
```

TODO: 补充DPO微调损失曲线


# 参考

本项目主要参考了以下项目：
- [ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [MINI_LLM](https://github.com/jiahe7ay/MINI_LLM)
- [llm-course](https://github.com/mlabonne/llm-course)