from collections import Counter
from typing import Union
from dataclasses import make_dataclass, field
from transformers import T5Config
import ctypes
import os
import platform
import re
import torch

from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import TrainingArguments, TrainerCallback

# from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import ujson


# 结束标点符号
END_PUN = set(".。!！）)》}】?？\"”")



# 保留中文和英文、下划线，不要标点符号
NON_CHAR = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")


def _get_doc_mini_hash(doc, num_perm: int) -> MinHash:
    '''
    获取一段文本的mini hash
    '''
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode('utf-8'))
    return mini_hash


class DropDatasetDuplicate:

    def __init__(self,  threshold: float = 0.85, num_perm: int = 256) -> None:
        '''
        获取一个数据集中所有重复（相似的超过threshold）的index，输入为：list[str]，一个str元素为一段文本(doc)
        如输入： [a, b, c, d, c, d, e] 返回：{4, 5} (后面两个 c, d 的index)
        '''
        self.similar_index_cluster = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def add_doc(self, index: object, doc: str,):
        '''
        添加文档，
        index： 文档的索引
        doc: 文档本身
        '''

        # 只保留中文和英文、下划线，不要标点符号
        doc = ''.join(NON_CHAR.split(doc))
        # doc = [''.join(t) for t in list(ngrams(doc, 3))]

        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close_duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        # 所有相似的doc在similar_index_cluster中的key都是最早出现的idx
        # 如：data中索引inndex 2, 7, 8, 9, 10, 12 是相似的，则在similar_index_cluster中表现为 {2: {8, 9, 10, 12}}
        if len(close_duplicates) > 0:
            min_idx = min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)

    def get_duplicate_indexs(self):
        '''
        返回所有的重复文档索引
        '''
        similar_index_cluster = self.similar_index_cluster
        need_to_remove_idx = set()

        for key_idx in similar_index_cluster.keys():
            need_to_remove_idx |= similar_index_cluster[key_idx]

        return need_to_remove_idx


def fixed_response(item: str) -> str:
    '''
    修复被截断的回答，从末尾往回找第一个结束标点
    '''
    if len(item) <= 1:
        return item
    if item[-1] in END_PUN:
        return item

    n = len(item)
    i = n - 1
    while i > 0 and item[i] not in END_PUN:
        i -= 1

    return ''.join(item[0: i + 1])


def fixed_space(sentence: str) -> str:
    '''单个空格删除，连续两个空格保留一个
    '''
    n = len(sentence)
    new_sentence = []
    i = 0
    while i < n:
        word = sentence[i]
        if word != ' ':
            new_sentence.append(word)
        elif i + 1 < n and sentence[i + 1] == ' ':
            new_sentence.append(word)
            i += 1  # 两个空格保留一个，指针往下走一步
        i += 1

    return ''.join(new_sentence)


def get_free_space_of_disk(folder: str = './') -> float:
    '''
    获取指定目录所在磁盘大小，返回单位: GB
    '''
    res_val = 0.0
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        res_val = free_bytes.value
    else:
        st = os.statvfs(folder)
        res_val = st.f_bavail * st.f_frsize

    return res_val / (1024 ** 3)


def my_average(arry_list) -> float:
    '''
    自定义均值计算，空数组返回0.0
    '''
    if len(arry_list) == 0:
        return 0.0

    return np.average(arry_list)


def json_to_dataclass(json_file: str, class_name: str = 'Config') -> type:
    '''
    将json配置文件转换为dataclass
    >>> example:
    >>> data_class = json_to_dataclass('my_config.json', 'Config')
    >>> my_config = data_class()
    >>> assert my_config.name == 'Alice'
    >>> my_config.name = 'Bob' 
    '''
    json_dict = {}
    with open(json_file, 'r', encoding='utf-8') as f:
        json_dict = ujson.load(f)

    # 将dict转换为可迭代的属性名称、属性类型，默认值
    fields_list = []
    for k, v in json_dict.items():
        fields_list.append((k, type(v), field(default=v)))

    data_class = make_dataclass(cls_name=class_name, fields=fields_list)

    return data_class


def get_path_of_suffix_files(root: str, suffix: str, with_create_time: bool = False) -> list:
    '''
        获取指定目录下下指定后缀的所有文件的绝对路径
    '''
    suffix_files = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                full_path = '{}/{}'.format(root, file)
                if with_create_time:
                    suffix_files.append(
                        (full_path, os.path.getctime(full_path)))
                else:
                    suffix_files.append(full_path)

    return suffix_files


def get_bleu4_score(reference, outputs, n_gram: int = 4) -> float:
    '''
    获取bleu4分数
    '''

    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not type(reference) is list:
        reference = list(reference)
    if not type(outputs) is list:
        outputs = list(outputs)

    outputs_counter = extract_Ngram(outputs, n_gram=n_gram)
    reference_counter = extract_Ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt

    for (key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt

    # print(clip_counter, output_ngram_counter)
    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter

    # bleu
    log_precision_scores = weights * np.log(precision_scores)

    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    # brevity_penalty = 1.0,   bleu = sentence_bleu([reference], outputs)
    # brevity_penalty = 1.0

    bleu = brevity_penalty * geometric_mean

    return bleu


def extract_Ngram(words_list, n_gram: int) -> tuple:
    '''
    获取一个句子的n_grama
    return：
        ngram_counter： key = ('w1  w2 ... wn', n_gram), value: count of key
    '''
    n = len(words_list)
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = ' '.join(words_list[j: j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter


def save_model_config(config_dict: dict, file: str) -> None:
    '''
    将模型配置写入到json文件, 输入模型保存的目录及文件名
    '''

    with open(file, 'w', encoding='utf-8') as f:
        ujson.dump(config_dict, f, indent=4, ensure_ascii=False)

