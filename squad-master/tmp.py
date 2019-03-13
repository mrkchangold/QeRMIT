import numpy as np
import os
import spacy
import ujson as json
import urllib.request
from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
import torch
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertModel
import collections
import pdb
from setup import convert_idx



def create_token2idx(context):
    token2idx = {}
    total_spaces = context.split()
    counter = 0
    for total_idx, word in enumerate(total_spaces):
        tokenized = tokenizer.tokenize(word)
        for token in tokenized:
            token2idx[counter] = total_idx
            counter += 1
    return token2idx


## processing starts here
# context_tokens_w_space would be context -> tokenized -> " ".join -> char idx -> context_tokens_w_space
def tokens2pred(context, spans, start_token, end_token, token2idx):
    total_spaces = context.split()
    context_tokens = tokenizer.tokenize(context)
    context_tokens_w_spaces = " ".join(context_tokens)

    pred = ''
    last = start_token
    word = context_tokens[start_token]
    print("starting word")
    print(word)
    for curr in range(start_token + 1,end_token + 1):
        if token2idx[curr] == token2idx[last]: # if they are from the same word
            stripped = context_tokens[curr].replace(" ##", "").replace("##", "").strip()
            print("stripped")
            print(stripped)
            word = word + stripped
        else:
            print("added word")
            print(word)
            pred = pred + ' ' + word
            word = context_tokens[curr]
            
        print(pred)
        
        last = curr
    pred = pred + ' ' + context_tokens[end_token]
    pred = pred.strip()
    return pred


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True) 

#get_final_text(pred_text = context_tokens_w_space, orig_text = total, do_lower_case = True, verbose_logging = False)
#"Hahaha Nice! Di's everchanging camera is named 'Charlie.' Oh, yes it is. BTW, I like your jeans"
total = "Hahaha Nice! Di's everchanging camera is named 'Charlie.' Oh, yes it is. BTW, I like your jeans"
#['ha', '##ha', '##ha', 'nice', '!', 'di', "'", 's', 'ever', '##chang', '##ing', 'camera', 'is', 'named', "'", 'charlie', '.', "'", 'oh', ',', 'yes', 'it', 'is', '.', 'bt', '##w', ',', 'i', 'like', 'your', 'jeans']
given = "Di's everchanging camera"
#['di', "'", 's', 'ever', '##chang', '##ing', 'camera', 'is', 'named', "'", 'charlie', '.', "'", 'oh', ',', 'yes', 'it', 'is', '.']
context_tokens = tokenizer.tokenize(total)
context_tokens_w_space = " ".join(str(x) for x in context_tokens)
spans = convert_idx(context_tokens_w_space, context_tokens)
#[(0, 2), (3, 4), (5, 6), (7, 11), (12, 19), (20, 25), (26, 32), (33, 35), (36, 41), (42, 43), (44, 51), (52, 53), (54, 55), (56, 58), (59, 60), (61, 64), (65, 67), (68, 70), (71, 72)]

start_token = 5
end_token = 11
token2idx = create_token2idx(total)

# assume we know the start and end idx of from the tokenized
# This is from start and end idx
orig_text = " ".join(total.split()[token2idx[start_token]:token2idx[end_token + 1]])

pred_text = tokens2pred(total, spans, start_token, end_token, token2idx)

final_text = get_final_text(pred_text, orig_text, do_lower_case = True, verbose_logging=False)

pdb.set_trace()

