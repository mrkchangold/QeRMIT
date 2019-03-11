import json
import os
import argparse
import csv
import run_squad
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertForQuestionAnswering2, BertForQuestionAnswering3, BertForQuestionAnswering_OG, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--json_file", default=None, type=str, help="predictions jsonfile location (output of run_squad). E.g., train-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--OG", action='store_true', help="test")

    args = parser.parse_args()

    with open(args.json_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

#    if not os.path.exists(args.output_dir):
#        os.makedirs(args.output_dir)
    train_examples = run_squad.read_squad_examples(args.json_file, is_training=True, version_2_with_negative=True)
    max_seq_len = 384
    max_query_len = 64
    max_answer_len = 30

    exceed_seq_lens = []
    exceed_query_lens = []
    exceed_answer_lens = []

    exceed_seq_len_counter = 0
    exceed_query_len_counter = 0
    exceed_answer_len_counter = 0
    overall_counter = 0

    max_s = 0
    max_q = 0
    max_a = 0
    
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True) # added_flag, currently hardcoded

    train_features = run_squad.convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=512,
        is_training=True)
    
    for example in train_features:
        overall_counter += 1
        if sum(example.input_mask) > max_seq_len:
            exceed_seq_lens.append(example.tokens)
            exceed_seq_len_counter += 1
        if sum(example.input_mask) > max_s:
            max_s = sum(example.input_mask)
        if sum(example.segment_ids_flipped) > max_query_len:
            exceed_query_lens.append(example.tokens)
            exceed_query_len_counter += 1
        if sum(example.segment_ids_flipped) > max_q:
            max_q = sum(example.segment_ids_flipped)
        if (example.end_position - example.start_position) > max_answer_len:
            exceed_answer_len_counter += 1
            exceed_answer_lens.append(example.tokens)
        if (example.end_position - example.start_position) > max_a:
            max_a = (example.end_position - example.start_position)
            
    print("Number of examples: %d." % overall_counter)
    print("Number of sequences that exceeded max_seq_len of %d is %d." % (max_seq_len, exceed_seq_len_counter))
    print("Number of queries that exceeded max_query_len of %d is %d." % (max_query_len, exceed_query_len_counter))
    print("Number of answers that exceeded max_answer_len of %d is %d." % (max_answer_len, exceed_answer_len_counter))
    print("Max seq length found was %d." % max_s)
    print("Max query length found was %d." % max_q)
    print("Max answer length found was %d." % max_a)


    #print(sum(train_features[0].input_mask))

if __name__ == "__main__":
    main()
