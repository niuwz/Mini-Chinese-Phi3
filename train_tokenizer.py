import os
import pandas as pd
# import sentencepiece as spm
import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast

from config import PROJECT_ROOT,DATA_ROOT,TEMP_ROOT


def check_dir_exits(dir: str) -> None:
    '''
    检查文件夹是否存在，如果不存在则创建文件夹
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def train_my_huggingface_wiki_tokenizer(cropus_file: str, max_train_line: int = None, vocab_size: int = 40960, token_type: str = 'char') -> None:
    '''
    训练tokenizer with huggingface
    '''

    tokenizer_slow_save_path = PROJECT_ROOT+"/tokenizer/"
    tokenizer_fast_save_path = PROJECT_ROOT+"/tokenizer/"

    check_dir_exits(PROJECT_ROOT+"/tokenizer/")
    check_dir_exits(tokenizer_fast_save_path)

    def get_training_corpus(buffer_size: int = 1000, chunk_len: int = 2048):
        '''
        一个文本块大小2048
        '''
        line_cnt = 0
        buffer = []
        with open(cropus_file, 'r', encoding='utf-8') as f_read:
            cur_chunk_txt, txt_len = [], 0
            for line in f_read:

                cur_chunk_txt.append(line)
                txt_len += len(line)
                line_cnt += 1

                if txt_len >= chunk_len:
                    buffer.append(
                        ''.join(cur_chunk_txt)
                    )
                    cur_chunk_txt, txt_len = [], 0

                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

                if isinstance(max_train_line, int) and line_cnt > max_train_line:
                    break

            # yield last
            if len(buffer) > 0:
                yield buffer

    special_tokens  = ["[PAD]", "[EOS]", "[BOS]"]

    if token_type == 'char':

        model = BPE(unk_token="[UNK]")
        tokenizer = Tokenizer(model)

        # 用兼容等价分解合并对utf编码进行等价组合，比如全角A转换为半角A
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])

        # 标点符号，数字，及Metaspace预分割（否则decode出来没有空格）
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [Punctuation(), Digits(individual_digits=True), Metaspace()]
        )

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.Metaspace()
    elif token_type == 'byte':

        # byte BPE n不需要unk_token
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True)

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.ByteLevel(
            add_prefix_space=False, use_regex=True)
        tokenizer.post_processor = tokenizers.processors.ByteLevel(
            trim_offsets=False)
    else:
        raise Exception(
            f'token type must be `char` or `byte`, but got {token_type}')

    trainer = BpeTrainer(vocab_size=vocab_size-2, min_frequency=100,
                         show_progress=True, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # add \t \n
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])

    tokenizer.save(tokenizer_slow_save_path)

    # 将训练的tokenizer转换为PreTrainedTokenizerFast并保存
    # 转换是为了方便作为`AutoTokenizer`传到其他`huggingface`组件使用。

    # 转换时要手动指定`pad_token`、`eos_token`等特殊token，因为它不指定你原来的tokenizer中哪些字符是这些特殊字符

    slow_tokenizer = tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=slow_tokenizer,
        pad_token="[PAD]",
        bos_token='[BOS]',
        eos_token='[EOS]',
    )

    fast_tokenizer.save_pretrained(tokenizer_fast_save_path)

    print(f'slow tokenizer save in path: {tokenizer_slow_save_path}')
    print(f'fast tokenizer save in path: {tokenizer_fast_save_path}')

    print(
        f"\ntrain tokenizer finished. you can use `AutoTokenizer.from_pretrained('{tokenizer_fast_save_path}')` to load and test your tokenizer.")


if __name__ == '__main__':
    data_path = DATA_ROOT + 'tokenizer_wiki.txt'
    vocab_size = 32000
    train_my_huggingface_wiki_tokenizer(data_path, vocab_size=vocab_size, token_type='byte')