"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from unilm.old_school_transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples, build_components
from unilm.old_school_transformers import \
    BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.tokenization_minilm import MinilmTokenizer

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'xlm-roberta': XLMRobertaTokenizer,
    'unilm': UnilmTokenizer,
}


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    config, tokenizer, model, bi_uni_pipeline, device = build_components(
        seed=args.seed,
        model_type=args.model_type,
        tokenizer_name=args.tokenizer_name,
        config_path=args.config_path,
        model_path=args.model_path,
        beam_size=args.beam_size,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_tgt_length=args.max_tgt_length,
        pos_shift=args.pos_shift,
        forbid_ignore_word=args.forbid_ignore_word,
        forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
        length_penalty=args.length_penalty,
        ngram_size=args.ngram_size,
        min_len=args.min_len,
        mode=args.mode,
        fp16=args.fp16,
    )
    model_recover_path = args.model_path.strip()

    next_i = 0
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    to_pred = load_and_cache_examples(
        args.input_file, tokenizer, local_rank=-1,
        cached_features_file=None, shuffle=False
    )

    input_lines = []
    for line in to_pred:
        input_lines.append(tokenizer.convert_ids_to_tokens(line["source_ids"])[:max_src_length])
    if args.subset > 0:
        logger.info("Decoding subset: %d", args.subset)
        input_lines = input_lines[:args.subset]

    input_lines = sorted(list(enumerate(input_lines)),
                         key=lambda x: -len(x[1]))
    output_lines = [""] * len(input_lines)
    score_trace_list = [None] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size)

    model.additional_scores_idx = {
        **{"max": []},
        **{idx: [] for idx in []}
    }

    with tqdm(total=total_batch) as pbar:
        batch_count = 0
        first_batch = True
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            next_i += args.batch_size
            batch_count += 1
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instances.append(proc(instance))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                traces = model(input_ids, token_type_ids,
                               position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (tokenizer.sep_token, tokenizer.pad_token):
                            break
                        output_tokens.append(t)
                    if args.model_type in ["roberta", "xlm-roberta"]:
                        output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                    else:
                        output_sequence = ' '.join(detokenize(output_tokens))
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
                    if args.need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
            pbar.update(1)
            first_batch = False

    if args.output_file:
        fn_out = args.output_file
    else:
        fn_out = model_recover_path+'.'+args.split
    with open(fn_out, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")

    if args.need_score_traces:
        with open(fn_out + ".trace.pickle", "wb") as fout_trace:
            pickle.dump(
                {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
            for x in score_trace_list:
                pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()

