from __future__ import absolute_import, division, print_function

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

###
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import csv # added_flag
from tensorboardX import SummaryWriter
import util
from ujson import load as json_load
# import setup_chris as setup
from collections import Counter
from run_squad import read_squad_examples

def tokenizer():
    train_examples = read_squad_examples(
        input_file='./data/train-v2.0.json', is_training=True, version_2_with_negative=True)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    
    for (example_index, example) in enumerate(train_examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        print(query_tokens)
        
        input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        print(input_ids)

if __name__ == '__main__':
    tokenizer()


    # features = []
    # for (example_index, example) in enumerate(examples):
    #     query_tokens = tokenizer.tokenize(example.question_text)

    #     if len(query_tokens) > max_query_length:
    #         query_tokens = query_tokens[0:max_query_length]

    #     tok_to_orig_index = []
    #     orig_to_tok_index = []
    #     all_doc_tokens = []
    #     for (i, token) in enumerate(example.doc_tokens):
    #         orig_to_tok_index.append(len(all_doc_tokens))
    #         sub_tokens = tokenizer.tokenize(token)
    #         for sub_token in sub_tokens:
    #             tok_to_orig_index.append(i)
    #             all_doc_tokens.append(sub_token)

    #     tok_start_position = None
    #     tok_end_position = None
    #     if is_training and example.is_impossible:
    #         tok_start_position = -1
    #         tok_end_position = -1
    #     if is_training and not example.is_impossible:
    #         tok_start_position = orig_to_tok_index[example.start_position]
    #         if example.end_position < len(example.doc_tokens) - 1:
    #             tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    #         else:
    #             tok_end_position = len(all_doc_tokens) - 1
    #         (tok_start_position, tok_end_position) = _improve_answer_span(
    #             all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
    #             example.orig_answer_text)

