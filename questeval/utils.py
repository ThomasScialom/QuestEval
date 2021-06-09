from typing import Dict, List, Optional, Tuple
import string
import re
import collections
import torch

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

class API_T2T:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_source_length: int,
        model_batch_size: int,
        keep_score_idx: int,  # Note: will work only if beamsize == 1
        device: str = "cuda"
    ) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        """
        self.model = CustomT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        """
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        self.keep_score_idx = keep_score_idx

        if device == "cuda":
            self.model.cuda()
        self.max_source_length = max_source_length
        self.model_batch_size = model_batch_size


    def predict(
        self,
        sources: List[str],
    ):
        # sources should be question <s> context

        gen_texts = []
        keep_score_idx_scores = []

        for i in range(0, len(sources), self.model_batch_size):
            inputs = self.tokenizer(
                sources[i: i+self.model_batch_size],
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                verbose=False,
            )

            with torch.no_grad():
                source_ids, source_mask = inputs["input_ids"], inputs["attention_mask"]
                dict_generated_ids = self.model.generate(
                    input_ids=source_ids.to(self.model.device),
                    attention_mask=source_mask.to(self.model.device),
                    use_cache=True,
                    decoder_start_token_id=None,
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                gen_text = self.tokenizer.batch_decode(
                    dict_generated_ids['sequences'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                gen_texts += gen_text
                keep_score_idx_scores += (1 - dict_generated_ids['scores'][0].softmax(-1)[:, self.keep_score_idx]).squeeze().tolist()

        # Note: self.model.additional_scores_idx keep in memory probs only if beam == 1;
        #   it is usefull only when T5 is used as a classifier so far.
        return keep_score_idx_scores, gen_texts


def calculate_f1_squad(
    a_gold: str,
    a_pred: str
) -> float:
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


def calculate_BERTScore(
    model_predictions: List[str],
    gold_references: List[str],
    metric_BERTScore,
    device: str,
) -> List[float]:

    if len(model_predictions) == 0:
        return []

    metric_BERTScore.add_batch(predictions=model_predictions, references=gold_references)
    final_score = metric_BERTScore.compute(model_type='bert-base-multilingual-cased', device=device)

    """
    # set all unanswerable scores to 0
    for i, (pred) in enumerate(model_predictions):
        if pred == "unanswerable":
            final_score['f1'][i] = 0.0
    """
    return [f1 for f1 in final_score['f1']]


def extract_table_answers(
    text: str
) -> List[str]:

    asws = []

    asw_toks = []
    is_asw = False
    for tok in text.split():

        if tok == ']':
            asws.append(' '.join(asw_toks))
            is_asw = False
            asw_toks = []

        if is_asw:
            asw_toks.append(tok)

        if tok == '[':
            is_asw = True
    return asws


def split_on_punct(
    doc
):
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


def sentencize(
    text: str,
    spacy_pipeline
) -> List:
    preprocessed_context = spacy_pipeline(text)
    return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]