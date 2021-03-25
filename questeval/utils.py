import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import string
import re
import collections

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import (
    T5ForConditionalGeneration,
    top_k_top_p_filtering,
    T5Tokenizer,
)


class PooledExamples:
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path,
        suffix,
        max_length,
        overwrite_cache=False,
        return_text=False,
        pool_size=400000,
        n_obs=None
    ):
        self.tokenizer = tokenizer
        self.data_dir, self.type_path = data_dir, type_path
        assert suffix in ["source", "target"], "Wrong suffix"
        self.suffix = suffix
        self.add_eos = (self.suffix == "target")
        self.max_length = max_length
        self.overwrite_cache = overwrite_cache
        self.return_text = return_text
        self.pool_size = pool_size
        self.n_obs = n_obs

        self.cached_file_paths, self.cached_file_lens = self._create_cached_files()

        self.position_pool_index = 0
        self.cached_file_path = ""
        self._examples = self._load_cached_file()

    @property
    def minimum_position_index(self):
        return (self.position_pool_index + 0) * self.pool_size

    @property
    def maximum_position_index(self):
        return (self.position_pool_index + 1) * self.pool_size

    def _create_cached_files(self) -> Tuple[List, List]:
        """
        We are going to create pools / cached files.
        """
        cached_file_paths, cached_file_lens = self._get_previously_cached_files()
        if len(cached_file_paths) > 0:
            return cached_file_paths, cached_file_lens

        data_path = os.path.join(self.data_dir, self.type_path + f".{self.suffix}")
        tok_name = "t5"
        cache_path = Path(f"{data_path}_{tok_name}{self.max_length}.pt")

        # Reading by batch of pool_size rows
        pool_index, lines = 0, None
        more_than_one_cached_file = False
        with open(data_path, "r") as f:
            # LOOP cached files of self.pool_size length
            while lines is None or len(lines) == self.pool_size:
                i, lines, line = 0, [], "temp"
                # LOOP rows of one cached file
                for _ in range(self.pool_size):
                    line = f.readline()
                    # EOF -> break
                    if len(line) == 0:
                        break
                    lines.append(line)

                    # max n_obs -> break
                    if self.n_obs is not None and (pool_index * self.pool_size + len(lines)) == self.n_obs:
                        break

                    # max pool_size -> break
                    if len(lines) == self.pool_size:
                        more_than_one_cached_file = True
                        break
                assert len(lines) > 0, f"Found empty file at {data_path}"

                # Tokenizing
                examples = []
                for text in tqdm(lines, desc=f"Tokenizing {self.type_path}.{self.suffix} (pool {pool_index})"):
                    original_text = text
                    if self.add_eos is True:
                        text += f" {self.tokenizer.eos_token}"
                    tokenized = self.tokenizer(
                        [text],
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        verbose=False,
                    )
                    assert tokenized.input_ids.shape[1] == self.max_length
                    if self.add_eos is True \
                            and tokenized["input_ids"][0, -1] not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                        tokenized["input_ids"][0, -1] = self.tokenizer.eos_token_id
                    if self.return_text is True:
                        tokenized["text"] = original_text
                    examples.append(tokenized)

                # Storing cached file (initial path OR new one with index of pool file)
                pool_cache_path = cache_path
                if more_than_one_cached_file is True:
                    pool_cache_path = Path(
                        str(pool_cache_path).replace(".pt", f"_pool{pool_index:03}.pt")
                    )
                torch.save([dict(ex) for ex in examples], pool_cache_path.open("wb"))

                cached_file_paths.append(pool_cache_path)
                cached_file_lens.append(len(examples))

                # max n_obs -> break
                if self.n_obs is not None and (pool_index * self.pool_size + len(lines)) == self.n_obs:
                    break

                pool_index += 1

        return cached_file_paths, cached_file_lens

    def _get_previously_cached_files(self) -> Tuple[List, List]:
        cached_file_paths, cached_file_lens = [], []
        tok_name = "t5"
        prefix = self.type_path + f".{self.suffix}_{tok_name}{self.max_length}"
        for filename in os.listdir(self.data_dir):
            if filename.startswith(prefix):
                cached_file_paths.append(Path(os.path.join(self.data_dir, filename)))

        cached_file_paths = sorted(cached_file_paths, key=lambda x: str(x))
        cached_file_lens = [len(torch.load(path)) for path in cached_file_paths]

        if len(cached_file_lens) > 0 and self.overwrite_cache is True:
            raise NotImplementedError

        if len(cached_file_paths) > 1 or (len(cached_file_lens) > 0 and self.pool_size < max(cached_file_lens)):
            self.pool_size = max(cached_file_lens)
            print(f"WARNING: pool_size has been changed to {self.pool_size} since previous files were cached.")

        return cached_file_paths, cached_file_lens

    def _load_cached_file(self)-> List:
        self.cached_file_path = self.cached_file_paths[self.position_pool_index]
        return torch.load(self.cached_file_path)

    def __len__(self):
        return sum(self.cached_file_lens)

    def __getitem__(self, item):
        if item < 0:
            item = self.__len__() + item

        # Case where not correct file is loaded in RAM
        if not (self.minimum_position_index <= item < self.maximum_position_index):
            self.position_pool_index = item // self.pool_size
            assert 0 <= self.position_pool_index < len(self.cached_file_paths), "Wrong position_pool_index"
            self._examples = self._load_cached_file()

        return self._examples[item - self.minimum_position_index]


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
        overwrite_cache=False,
        return_index=False,
        return_text=False,
        pool_size=400000,
    ):
        super(SummarizationDataset, self).__init__()

        pooled_exmaples_kwargs = {
            "tokenizer": tokenizer,
            "data_dir": data_dir,
            "type_path": type_path,
            "overwrite_cache": overwrite_cache,
            "return_text": return_text,
            "pool_size": pool_size,
            "n_obs": n_obs,
        }
        self.source = PooledExamples(
            suffix="source", max_length=max_source_length,
            **pooled_exmaples_kwargs
        )
        self.target = PooledExamples(
            suffix="target", max_length=max_target_length,
            **pooled_exmaples_kwargs
        )

        self.pad_token_id = tokenizer.pad_token_id
        self.return_index = return_index
        self.return_text = return_text

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()

        res = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }
        if self.return_index is True:
            res.update({"index": index})

        if self.return_text is True:
            res.update({"target_text": self.target[index]["text"]})
            res.update({"source_text": self.source[index]["text"]})

        return res


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
class T2TDataCollator:
    def __init__(self, tokenizer, mode='training'):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        target_ids = torch.stack([example['decoder_input_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])

        pad_token_id = self.tokenizer.pad_token_id

        input_ids, attention_mask = self._trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
        target_ids = self._trim_batch(target_ids, pad_token_id)

        lm_labels = target_ids.clone()
        decoder_input_ids = None
        if self.mode == 'training':
            lm_labels[lm_labels[:, :] == pad_token_id] = -100

        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
        }

        if "index" in batch[0]:
            params.update({"indexs": [example['index'] for example in batch]})

        if "source_text" in batch[0] and "target_text" in batch[0]:
            params.update({"source_text": [example['source_text'] for example in batch]})
            params.update({"target_text": [example['target_text'] for example in batch]})

        if decoder_input_ids is not None:
            params["decoder_input_ids"] = decoder_input_ids

        return params

    @staticmethod
    def _trim_batch(
        input_ids, pad_token_id, attention_mask=None,
    ):
        """Remove columns that are populated exclusively by pad_token_id"""
        keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
        if attention_mask is None:
            return input_ids[:, keep_column_mask]
        else:
            return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    no_init_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "Initialisation random of the weights (i.e. not the learned one via self supervision)"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "Path for data train / eval / test"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

    from_slurm: Optional[int] = field(
        default=0,
        metadata={"help": "Are we working from slurm or not"},
    )
    eval_output: bool = field(
        default=False,
        metadata={"help": "Do we need a eval output file with model predictions ?"},
    )
    eval_output_name: str = field(
        default="",
        metadata={"help": "You can add a particular name for this eval."},
    )


def split_on_punct(doc):
    """
    From one spacy doc to a List of (sentence_text, (start, end))
    """
    start = 0
    seen_period = False
    start_idx = 0
    for i, token in enumerate(doc):
        if seen_period and not token.is_punct:
            yield doc[start: token.i].text, (start_idx, token.idx)
            start = token.i
            start_idx = token.idx
            seen_period = False
        elif token.text in [".", "!", "?"]:
            seen_period = True
    if start < len(doc):
        yield doc[start: len(doc)].text, (start_idx, len(doc.text))


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    """
    In order to retrieve confidence score for token "unanswerable" at inference,
    we built this class. It corresponds to the token "__un" of the vocabulary (
    first word pieced token of word "unanswerable".

    changes at rows:
    "# all scores should be empty at this point. (...)"
    "# stack the memory scores (...)"
    "# Keep in memory generation scores (...)"
    """

    def __init__(self, *args, **kwargs):
        self.additional_scores_idx = {}
        super(CustomT5ForConditionalGeneration, self).__init__(*args, **kwargs)

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = (encoder_outputs, None) if encoder_outputs is not None else None

        temp_additional_scores_idx = {}
        for idx in self.additional_scores_idx.keys():
            temp_additional_scores_idx[idx] = []

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )

            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

                # Keep in memory generation scores
                if temp_additional_scores_idx != {}:
                    soft_probs = torch.softmax(next_token_logits, 1)
                    for idx in temp_additional_scores_idx:
                        if idx == "max":
                            idx_probs = torch.topk(soft_probs, 1)[0].squeeze()
                        else:
                            idx_probs = soft_probs[:, idx]
                        temp_additional_scores_idx[idx].append(idx_probs.cpu())


            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # stack the memory scores
        if not do_sample:
            for idx in self.additional_scores_idx:
                # if batch_size == 1 we unsqueeze it to unified list format
                if len(temp_additional_scores_idx[idx][0].size()) == 0:
                    for i, item in enumerate(temp_additional_scores_idx[idx]):
                        temp_additional_scores_idx[idx][i] = item.unsqueeze(0)

                self.additional_scores_idx[idx] += torch.stack(temp_additional_scores_idx[idx], -1).tolist()

        return input_ids


class APIQA:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_source_length: int,
        model_batch_size: int,
        keep_score_idx: list = [],  # Note: will work only if beamsize == 1
        device: str = "cuda"
    ) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = CustomT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        self.top_p = 1
        self.keep_score_idx = keep_score_idx

        if device == "cuda":
            self.model.cuda()
        self.max_source_length = max_source_length
        self.model_batch_size = model_batch_size

    def _initialize_model_additional_scores_idx(self) -> None:
        self.model.additional_scores_idx = {
            **{"max": []},
            **{idx: [] for idx in self.keep_score_idx}
        }

    def predict(
        self,
        sources: List[str],
        beam_size: Optional[int] = None,
    ):
        # sources should be question <s> context
        gen_texts = []
        self._initialize_model_additional_scores_idx()
        for i in range(0, len(sources), self.model_batch_size):
            inputs = self.tokenizer(
                sources[i: i+self.model_batch_size],
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                verbose=False,
            )

            if self.top_p < 1:
                do_sample = True
                num_beams = 1
                num_return_sequences = beam_size
            else:
                do_sample = False
                num_beams = beam_size
                num_return_sequences = beam_size

            with torch.no_grad():
                source_ids, source_mask = inputs["input_ids"], inputs["attention_mask"]
                generated_ids = self.model.generate(
                    input_ids=source_ids.to(self.model.device),
                    attention_mask=source_mask.to(self.model.device),
                    use_cache=True,
                    decoder_start_token_id=None,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    top_p=self.top_p,
                    do_sample=do_sample
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                gen_texts += gen_text

        # Note: self.model.additional_scores_idx keep in memory probs only if beam == 1;
        #   it is usefull only when T5 is used as a classifier so far.
        return self.model.additional_scores_idx, gen_texts


def calculate_f1_squad(a_gold: str, a_pred: str) -> float:
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(s):
        if not s: return []
        return normalize_answer(s).split()

    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
