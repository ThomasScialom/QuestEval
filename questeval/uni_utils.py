from typing import List, Optional
import torch
from s2s_ft.utils import build_components
import s2s_ft.s2s_loader as seq2seq_loader


class T2tUniModel:
    def __init__(
        self,
        path_model,
        isCuda,
        keep_score_idx=[],
        model_batch_size=48,
    ):
        beam_size = 1
        config, tokenizer, model, bi_uni_pipeline, device = build_components(
            model_type="xlm-roberta",
            tokenizer_name="xlm-roberta-base",
            model_path=path_model,
            beam_size=beam_size,
            max_seq_length=512,
            max_tgt_length=48,
            forbid_ignore_word=".",
            forbid_duplicate_ngrams=True,
            length_penalty=0.0,
            ngram_size=3,
            min_len=0,
            mode="s2s",
            use_cuda_if_available=isCuda,
        )

        self.keep_score_idx = keep_score_idx
        self.batch_size = model_batch_size

        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.bi_uni_pipeline = bi_uni_pipeline
        assert len(self.bi_uni_pipeline) == 1, "Not usual pipeline."
        self.device = device

    def _initialize_model_additional_scores_idx(self) -> None:
        # tokenizer.tokenize("unanswerable")
        # ['▁una', 'ns', 'wer', 'able']
        # tokenizer.convert_tokens_to_ids(['▁una', 'ns', 'wer', 'able'])
        # [220, 1779, 6488, 2886]
        self.model.additional_scores_idx = {
            **{"max": []},
            **{idx: [] for idx in self.keep_score_idx}
        }

    def predict(self, inputs: List[str], beam_size: Optional[int] = 1):
        if beam_size > 1:
            print("beam_size should be at 1 !")
            exit(-1)

        self._initialize_model_additional_scores_idx()

        # TODO: inputs should be sorted by length in order to reduce computation.
        # Tokenizing as wp, processing through pipeline
        processed_inputs = []
        for i in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[i: i + self.batch_size]

            # Tokenize
            tokenized_batch = []
            for example in batch_inputs:
                tokenized_batch.append(self.tokenizer.tokenize(example))
            max_local_len = max([len(x) for x in tokenized_batch])

            if max_local_len > 512 - 2 - 20:
                max_local_len = 512 - 2 - 20
            # To ids + padding to local_max_length
            for tokenized_input in tokenized_batch:

                tokenized_input = tokenized_input[:512-2 - 20]
                processed_input = self.bi_uni_pipeline[0](
                    (tokenized_input, max_local_len)
                )
                processed_inputs.append(processed_input)

        # Batching then predicting
        output_sequences = []
        for i in range(0, len(processed_inputs), self.batch_size):
            batch_inputs = processed_inputs[i: i+self.batch_size]

            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(batch_inputs)
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = [
                    t.to(self.device) if t is not None else None for t in batch
                ]
                traces = self.model(
                    input_ids, token_type_ids, position_ids, input_mask,
                    task_idx=task_idx, mask_qkv=mask_qkv
                )

                output_ids = traces.tolist()

                # De-tokenizing
                for output in output_ids:
                    output_buf = self.tokenizer.convert_ids_to_tokens(output)
                    output_tokens = []
                    for t in output_buf:
                        if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                            break
                        output_tokens.append(t)
                    output_sequence = self.tokenizer.convert_tokens_to_string(output_tokens)
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))

                    output_sequences.append(output_sequence)

        return self.model.additional_scores_idx, output_sequences
