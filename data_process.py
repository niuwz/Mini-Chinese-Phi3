from utils.functions import get_path_of_suffix_files, DropDatasetDuplicate
from config import PROJECT_ROOT, DATA_ROOT, TEMP_ROOT
from logger import Logger
import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import codecs
import csv
import pandas as pd
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
import pyarrow.parquet as pq
# from opencc import OpenCC

import sys
sys.path.extend(['.', '..'])


log = Logger('data_process', save2file=True,
             file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

punctuation = set(
    "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："


def delete_file(file: str) -> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False


def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence)

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans


def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence


def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len = len(set_a) + len(set_b)

    if total_len == 0:
        return 0.0

    inter_set = set_a & set_b

    return (2 * len(inter_set)) / total_len


def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True

    write(file_name, data_frame, compression='GZIP', append=append)


def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int = 10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数
    如：
    >>> def call_back(inputs: str) -> dict:
    >>>     if check(inputs) not valid:
    >>>         return None
    ...    
    ...    do something for inputs
    ...
    >>>     my_dict = {
    >>>             'prompt': inputs['p'],
    >>>             'response': inputs['a1'] + inputs['a2'],
    >>>             ...
    >>>         }
    >>>     return my_dict
    '''

    log.info('process file:{}'.format(read_file), save_to_file=True)
    start = time.time()

    raw_line_cnt = 0
    keep_line_cnt = 0

    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                raw_line_cnt += 1

                write_dict = call_back(line)

                if write_dict is None:
                    continue

                keep_line_cnt += 1
                append(write_dict)
                # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                # ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                # f_write.write('\n')

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e

        # end for
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []

    end = time.time()

    log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'
             .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start), save_to_file=True)


# =====================================数据集处理=================================

def process_bake_qa(response_less_word: int = 15, max_length: int = 512, group_cnt: int = 10000) -> None:
    '''
    处理147万百度知道知识类数据集

    '''
    file_names = [
        DATA_ROOT + 'raw_data/baike/baike_qa_train.json',
        DATA_ROOT + 'raw_data/baike/baike_qa_valid.json',
    ]

    save_file_name = PROJECT_ROOT + '/datasets/processed_data/baike_qa.parquet'
    # 后续append写入，存在文件先删除
    if exists(save_file_name):
        assert delete_file(save_file_name)

    def find_previous(line1, line2):
        punctuates = ["！", "。", "？"]
        ridx = max([line1.rfind(i) for i in punctuates])
        if ridx < max_length//2:
            return line2[:max_length], line2[max_length:]
        return line1[ridx+1:]+line2[:ridx+1], line2[ridx+1:]

    def precess_line(item):
        if len(item['answer']) < response_less_word:
            return ""
        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item['title'], item['desc']) >= 0.90:
            # title 和desc 相似度过高，只用title作为问题
            prompt = item['title']
        else:
            # title 和desc拼接形成问题
            prompt = "{}{}".format(item['title'], item['desc'])

        # 删除\r
        prompt = prompt.replace('\r', '')

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = item['answer'].replace('\r', '')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return ""
        return prompt + response
    shufix = ".json"
    # for file_name in file_names:
    #     read_file = PROJECT_ROOT + file_name
    cur_rows = []
    save_path = PROJECT_ROOT + '/datasets/processed_data/'
    append = cur_rows.append
    all_cnt, keep_cnt = 0, 0
    for file in file_names:
        save_file = save_path + file.split("/")[-1][:-len(shufix)] + ".parquet"
        print(save_file)
        if exists(save_file):
            assert delete_file(save_file)
        with open(PROJECT_ROOT+file, "r", encoding="utf-8") as f:
            data = [ujson.loads(line) for line in f]
        log.info("process file: {}".format(file), save_to_file=True)
        for line in progress.track(data):
            all_cnt += 1
            line = precess_line(line)
            if len(line) < response_less_word:
                continue
            keep_cnt += 1
            line1 = ""
            while len(line) >= max_length:
                line1, line = find_previous(line1, line)
                write_dict = {
                    "text": line1
                }
                append(write_dict)
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append
            if len(line) < max_length:
                if len(line1) > 0:
                    line, _ = find_previous(line1, line)
                append({"text": line})
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append
 # end for
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(
        save_file, all_cnt, keep_cnt), save_to_file=True)

    # read_and_write_template(read_file, save_file_name, process_function)


def repair_line_error_csv_file(raw_csv_file: str, save_suffix: str, read_encoding: str = 'utf-8', ) -> None:
    '''
        修复csv文件，将文件中换行符替换为\n，字段中的英文字符替换为中文字符
    '''

    with codecs.open(raw_csv_file, 'r', encoding=read_encoding, errors='ignore') as f:
        reader = csv.reader(f)
        new_lines = []

        for line in reader:
            for i in range(len(line)):
                line[i] = line[i].replace('\n', '\\n')  # 处理异常的换行符
                line[i] = line[i].replace(',', '，')  # 英文逗号换为中文逗号
            new_lines.append(line)

        with open(raw_csv_file[: -4] + save_suffix, 'w', encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_lines)


def process_zhihu_kol_dataset(prompt_less_word: int = 4, response_less_word: int = 10, group_cnt: int = 10000, max_length: int = 512) -> None:
    '''
    处理知乎数据集
    '''

    raw_zhihu_data_path = abspath(
        dirname(__file__)) + '/datasets/raw_data/zhihu/'
    file_names = []
    suffix = '.parquet'
    for root, _, files in walk(raw_zhihu_data_path):
        for file in files:
            if file.endswith(suffix):
                file_names.append(root + '/' + file)

    def process_function(sentence: str) -> str:
        '''
        针对一个句子的数据清洗
        '''
        # 删除\r
        sentence = sentence.replace('\r', '')

        # 删除重复的标点符号
        sentence = remove_duplicate_punctuation(sentence)

        return sentence

    def find_previous(line1, line2):
        punctuates = ["！", "。", "？"]
        ridx = max([line1.rfind(i) for i in punctuates])
        if ridx < max_length//2:
            return line2[:max_length], line2[max_length:]
        return line1[ridx+1:]+line2[:ridx+1], line2[ridx+1:]

    # row keys :['INSTRUCTION', 'RESPONSE', 'SOURCE', 'METADATA']
    save_file = PROJECT_ROOT + '/datasets/processed_data/zhihu_kol.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file):
        assert delete_file(save_file)

    all_cnt, keep_cnt = 0, 0
    cur_rows = []
    append = cur_rows.append
    for file in file_names:
        pf = pq.read_table(file)
        log.info('process file: {}'.format(file), save_to_file=True)

        for prompt, response in progress.track(zip(pf['INSTRUCTION'], pf['RESPONSE']), total=pf.num_rows):
            all_cnt += 1
            prompt, response = prompt.as_py(), response.as_py()

            # prompt = process_function(prompt)
            # response = process_function(response)

            if len(prompt) < prompt_less_word or len(response) < response_less_word:
                continue
            line = process_function(prompt + response)

            line1 = ""
            while len(line) >= max_length:
                line1, line = find_previous(line1, line)
                write_dict = {
                    "text": line1
                }
                append(write_dict)
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append
            if len(line) < max_length:
                if len(line1) > 0:
                    line, _ = find_previous(line1, line)
                append({"text": line})
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append

            # if len(prompt) < prompt_less_word or len(response) < response_less_word:
            #     continue

            # keep_cnt += 1
            # write_dict = {
            #     "text": prompt+response
            #     # 'prompt': prompt,
            #     # 'response': response,
            # }
            # append(write_dict)

            # if len(cur_rows) >= group_cnt:
            #     df = pd.DataFrame(cur_rows)
            #     write_single_parquet_file(save_file, df)
            #     cur_rows = []
            #     append = cur_rows.append

    # end for
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(
        save_file, all_cnt, keep_cnt), save_to_file=True)


def process_sky_dataset(least_word: int = 10, max_length: int = 256, group_cnt: int = 10000):
    """
    skywork
    """
    # data_path = PROJECT_ROOT + '/datasets/raw_data/sky/'
    # save_path = PROJECT_ROOT + '/datasets/processed_data/'
    data_path = DATA_ROOT
    save_path = DATA_ROOT
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    def precess_line(line: str) -> str:
        line = line.strip()
        line = line.replace('\n', '')
        line = line.replace("\r", "")
        line = remove_duplicate_punctuation(line)
        return line

    def find_previous(line1, line2):
        punctuates = ["！", "。", "？"]
        ridx = max([line1.rfind(i) for i in punctuates])
        if ridx < max_length//2:
            return line2[:max_length], line2[max_length:]
        return bos_token+line1[ridx+1:]+line2[:ridx+1], line2[ridx+1:]
    shufix = ".jsonl"
    file_names = []
    for root, _, files in walk(data_path):
        for file in files:
            if file.endswith(shufix):
                if file.split("_")[-1] > "0004.jsonl":
                    file_names.append(root + '/' + file)
    print(file_names)
    cur_rows = []
    append = cur_rows.append
    all_cnt, keep_cnt = 0, 0
    for file in file_names:
        save_file = save_path + file.split("/")[-1][:-len(shufix)] + ".parquet"
        print(save_file)
        if exists(save_file):
            assert delete_file(save_file)
        with open(file, "r", encoding="utf-8") as f:
            data = [ujson.loads(line) for line in f]
        log.info("process file: {}".format(file), save_to_file=True)
        for line in progress.track(data):
            all_cnt += 1
            line = precess_line(line["text"]) + eos_token
            if len(line) < least_word:
                continue
            keep_cnt += 1
            line1 = ""
            while len(line) >= max_length:
                line1, line = find_previous(line1, line)
                write_dict = {
                    "text": line1
                }
                append(write_dict)
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append
            if len(line) < max_length:
                if len(line1) > 0:
                    line, _ = find_previous(line1, line)
                append({"text": line})
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file, df)
                    cur_rows = []
                    append = cur_rows.append
 # end for
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    remove_dataset_duplicate_rows(save_file, save_file)
    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(
        save_file, all_cnt, keep_cnt), save_to_file=True)


def remove_dataset_duplicate_rows(from_parquet_files, save_file, groups_cnt: int = 50000) -> None:
    '''
    使用mini_hash删除数据集中重复的部分
    '''
    # from_parquet_files = PROJECT_ROOT + '/datasets/dataset.parquet'
    #
    # save_file = PROJECT_ROOT + '/datasets/dataset_no_dulpticates.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file):
        assert delete_file(save_file)

    cur_rows = []
    all_cnt, keep_cnt = 0, 0
    row_index = -1
    drop_dataset_duplicate = DropDatasetDuplicate(threshold=0.85, num_perm=256)

    parquet_table = pq.read_table(from_parquet_files)
    all_cnt = parquet_table.num_rows

    # 先顺序遍历获取哪些行是重复的
    for doc in progress.track(parquet_table['text'], total=parquet_table.num_rows):

        row_index += 1
        drop_dataset_duplicate.add_doc(index=row_index, doc=str(doc))

    row_index = -1
    need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()

    # 再顺序遍历一遍，重复的行不添加到新的数据集
    for text in progress.track(parquet_table['text'], total=parquet_table.num_rows):
        row_index += 1  # 不管有没有跳过行, row_index都必须+1

        # 重复的行跳过
        if row_index in need_to_drop_indexs:
            continue

        cur_rows.append({'text': text.as_py()})
        keep_cnt += 1

        if len(cur_rows) >= groups_cnt:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file, df)
            cur_rows = []

    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)

    log.info("merge into file: {}, 全部数据共{}行，文档去重后剩余{}行".format(
        save_file, all_cnt, keep_cnt), save_to_file=True)


def process_belle_knowledge_enhanced_dataset(response_less_words: int = 15, group_cnt: int = 10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        DATA_ROOT + 'train_2M_CN.json',
        DATA_ROOT + 'Belle_open_source_1M.json',
    ]

    save_file = DATA_ROOT + '/sft_data/belle_3M_cn.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file):
        assert delete_file(save_file)

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        instruction = item['instruction']
        output = item['output']

        # 剔除翻译任务
        if '翻译' in instruction or 'translate' in instruction.lower():
            return None

        # 删除表格类任务
        if '表格' in instruction or '-----' in instruction or '-----' in output:
            return None

        if len(output) < response_less_words:
            return None

        instruction = remove_duplicate_punctuation(instruction)
        output = remove_duplicate_punctuation(output)

        if len(output) < response_less_words:
            return None

        write_dict = {
            'instruction': instruction,
            'output': output
        }

        return write_dict

    for file in file_names:
        read_and_write_template(file, save_file, process_function)


if __name__ == '__main__':

    # processed_file_dir = PROJECT_ROOT + '/datasets/processed_data'
    # if not exists(processed_file_dir):
    #     mkdir(processed_file_dir)

    # process_sky_dataset()
    process_belle_knowledge_enhanced_dataset(response_less_words=10)
    
