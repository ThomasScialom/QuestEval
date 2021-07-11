from typing import List, Tuple, Dict, Callable
import os
import json
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
from questeval import DIR, __version__
from questeval.utils import (
    API_T2T,
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers,
    text2hash
)

HF_ORGANIZATION = "ThomasNLG"

class QuestEval:
    def __init__(
        self,
        task: str = "text2text",
        language: str = "en",
        answer_types: Tuple = ('NER', 'NOUN'),
        list_scores: Tuple = ('answerability', 'bertscore', 'f1'),
        src_preproc_pipe=None,
        do_weighter: bool = False,
        do_consistency: bool = False,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        reduction_multi_refs: Callable = max,
        no_cuda: bool = False,
        use_cache: bool = True
    ) -> None:
        """
        Main class for the QuestEval metric

        Args:
            task (:str):
                the task to evaluate with QuestEval

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        """
        format for the json logs:
            hash(txt) #json file name
                {
                'type': type
                'text': text or pointer if image
                'self': {answer_type_1: {'answers': [answers],
                                         {'model_QG_1': {'questions': [questions],
                                                         'model_Weighter_1': [weights]
                                                        }
                                         }
                        }
                'asked': {question_1: {model_QA_1:  'answer': answer,
                                                    'answerability': score
                                                    'ground_truth': {answer_1: {'f1': score,
                                                                              'bertscore: score}
                                    }
                        }
                }
        """
        self.AVAILABLE_LANGUAGES = ("en",)  # todo: "multi"
        self.AVAILABLE_TASKS = ("text2text", "summarization", "text_simplification", "data2text")

        if task not in self.AVAILABLE_TASKS:
            logging.warning(f"Task {task} is not known. Setting the default text2text task. ")
            task = "text2text"

        if language not in self.AVAILABLE_LANGUAGES:
            raise (
                f"Language {language} is not implemented. The list of available languages are: {self.AVAILABLE_LANGUAGES}."
            )

        if task == 'summarization' and do_weighter is False:
            logging.warning(
                "Task is summarization but the weighter is deactivate. Set do_weighter=True to activate it when loading QuestEval."
            )

        self.log_dir = os.path.join(DIR, 'logs')
        self.hash_files = set(os.listdir(self.log_dir))
        self.use_cache = use_cache

        self.task = task
        self.language = language

        self.answer_types = answer_types
        self.src_preproc_pipe = src_preproc_pipe
        self.limit_sent = limit_sent
        self.sep = "</s>"
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self.reduction_multi_refs = reduction_multi_refs
        self.do_consistency = do_consistency
        self.do_weighter = do_weighter
        self.list_scores = list_scores
        if 'bertscore' in self.list_scores:
            self.metric_BERTScore = load_metric("bertscore")

        if language == 'en':
            try:
                self.spacy_pipeline = spacy.load('en_core_web_sm')
            except OSError:
                logging.warning("Downloading language model for the spaCy model.")
                from spacy.cli import download
                download('en_core_web_sm')
                self.spacy_pipeline = spacy.load('en_core_web_sm')

        if self.src_preproc_pipe is None:
            if task == 'data2text':
                """
                structured tables should be linearized for our QA/QG format this way:
                'name [ The Eagle ] , eatType [ coffee shop ] , food [ French ] , priceRange [ £ 2 0 - 2 5 ] , customer rating [ 3 out of 5 ] , area [ city centre ] , familyFriendly [ yes ] , near [ Burger King ]'
                we handle by default the preprocessing of the table for webnlg given the GEM format (https://gem-benchmark.com/)
                if your tables are in an other format, please pass a custom function for src_preproc_pipe
                """
                from questeval.utils import LinearizeWebnlgInput
                self.src_preproc_pipe = LinearizeWebnlgInput(spacy_pipeline=self.spacy_pipeline)

        logging.info("Loading the models, it can take time to download at first time.")
        self.models = self._load_all_models()

    def _load_all_models(self) -> Dict:
        # Textual hypothesis
        models = {"hyp": {}}
        if self.language == 'en':
            models['hyp']['QA'] = f'{HF_ORGANIZATION}/t5-qa_squad2neg-en'
            models['hyp']['QG'] = f'{HF_ORGANIZATION}/t5-qg_squad1-en'
        else:
            raise("Multilingual evaluation not handled yet.")

        # (if) multimodal sources
        if self.task == "data2text":
            models['src'] = dict()
            models['src']['QA'] = f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en'
            models['src']['QG'] = f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en'

        # Loading all the different models
        for modality in models.keys():
            for task in models[modality].keys():
                if not type(models[modality][task]) == str:
                    continue
                models[modality][task]= self.get_model(model_name=models[modality][task])

        # Loading the weighter
        models['Weighter'] = None
        if self.do_weighter:
            models['Weighter'] = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-weighter_cnndm-en')

        # Linking already loaded models for the other keys
        for k in ["src", "ref"]:
            if models.get(k) == None:
                models[k] = dict()
                models[k]['QA'] = models['hyp']['QA']
                models[k]['QG'] = models['hyp']['QG']

        return models

    def corpus_questeval(
        self,
        hypothesis: List[str],
        sources: List[str] = None,
        list_references: List[List[str]] = None,
        batch_size: int = 512
    ) -> Dict:

        assert hypothesis is not None

        having_sources = (
            sources is not None
            and all([isinstance(s, str) for s in sources])  # Only str allowed
        )
        having_references = (
            list_references is not None
            and all([isinstance(r, str) for rs in list_references for r in rs])  # Only str allowed
            and len(set([len(rs) for rs in list_references])) == 1  # Same number of refs per ex
        )

        assert having_sources or having_references, "You need to provide at least correct sources or correct references."
        if having_references:
            assert len(list_references) == len(hypothesis)
        if having_sources:
            assert len(sources) == len(hypothesis)

        scores = []
        for ex_idx in range(0, len(hypothesis), batch_size):
            logging.info(f"Total examples: {len(hypothesis)}. Proceeding the examples {ex_idx}")
            batch_sources, batch_list_references = None, None
            if having_sources:
                batch_sources = sources[ex_idx:ex_idx + batch_size]
            if having_references:
                batch_list_references = list_references[ex_idx:ex_idx + batch_size]
            scores += self._batch_questeval(
                hypothesis=hypothesis[ex_idx:ex_idx + batch_size],
                sources=batch_sources,
                list_references=batch_list_references,
            )

        result = {'corpus_score': np.average(scores), 'ex_level_scores': scores}
        return result

    def _batch_questeval(
        self,
        hypothesis: List[str],
        sources: List[str] = None,
        list_references: List[List[str]] = None,
    ) -> List[float]:

        list_compared_logs = []
        d_loaded_logs = dict()

        # Hypothesis
        hyp_logs, hyp_hashes, modified_logs = self._texts2logs(hypothesis, type_logs='hyp', d_loaded_logs=d_loaded_logs)
        if modified_logs:
            self._serialize_logs(hyp_logs, hyp_hashes)

        # Source
        if sources is not None:
            src_logs, src_hashes, modified_logs = self._texts2logs(sources, type_logs='src', d_loaded_logs=d_loaded_logs)
            # Asking the questions on the compared text
            modified_logs = max(self._compute_question_answering(src_logs, hyp_logs, 'src', 'hyp'), modified_logs)
            modified_logs = max(self._compute_question_answering(hyp_logs, src_logs, 'hyp', 'src'), modified_logs)
            # Compute the similarity scores
            modified_logs = max(self._compute_answer_similarity_scores(src_logs, type_logs='src'), modified_logs)
            # Serialise logs
            if modified_logs:
                self._serialize_logs(src_logs, src_hashes)
                self._serialize_logs(hyp_logs, hyp_hashes)
            list_compared_logs.append(src_logs)

        # Reference
        if list_references is not None:
            len_refs = [len(refs) for refs in list_references]
            assert min(len_refs) == max(len_refs), \
                "The number of references used to compute the score among the example should  be consistant."
            for i_ref in range(len_refs[0]):
                references = [refs[i_ref] for refs in list_references]
                ref_logs, ref_hashes, modified_logs = self._texts2logs(references, type_logs='ref', d_loaded_logs=d_loaded_logs)
                # Asking the questions on the compared text
                modified_logs = max(self._compute_question_answering(ref_logs, hyp_logs, 'ref', 'hyp'), modified_logs)
                modified_logs = max(self._compute_question_answering(hyp_logs, ref_logs, 'hyp', 'ref'), modified_logs)
                # Compute the similarity scores
                modified_logs = max(self._compute_answer_similarity_scores(ref_logs, type_logs='ref'), modified_logs)
                # Serialise logs
                if modified_logs:
                    self._serialize_logs(ref_logs, ref_hashes)
                    self._serialize_logs(hyp_logs, hyp_hashes)
                list_compared_logs.append(ref_logs)

        # Compute the similarity scores for hyp
        modified_logs = self._compute_answer_similarity_scores(hyp_logs, type_logs='hyp')
        # Serialise hyp logs
        if modified_logs:
            self._serialize_logs(hyp_logs, hyp_hashes)

        list_compared_logs = [
            [
                list_compared_logs[i][j]
                for i in range(len(list_compared_logs))
            ]
            for j in range(len(list_compared_logs[0]))
        ]

        # Calculate Score
        scores = []
        for hyps_log, compared_logs in zip(hyp_logs, list_compared_logs):
            scores.append(self._calculate_score_from_logs(hyps_log, compared_logs))

        return scores

    def _texts2logs(
        self,
        texts: List[str],
        type_logs: str,
        d_loaded_logs: Dict
    ):
        modified_logs = False

        # Preprocessing
        if type_logs == 'src' and self.src_preproc_pipe is not None:
            texts = [self.src_preproc_pipe(source) for source in texts]

        logs, logs_hashes = self._load_logs(texts, type_logs, d_loaded_logs)
        # Selecting the answers
        modified_logs = max(self._compute_answer_selection(logs, type_logs), modified_logs)
        #  Generating the questions
        modified_logs = max(self._compute_question_generation(logs, type_logs), modified_logs)
        # Asking the questions on itself (Round trip consistency)
        if self.do_consistency:
            modified_logs = (self._compute_question_answering(logs, logs, type_logs, type_logs), modified_logs)
        # Weighter
        if type_logs == 'src' and self.do_weighter:
            modified_logs = max(self._compute_weighter(logs, type_logs='src'), modified_logs)

        return logs, logs_hashes, modified_logs

    def _load_logs(
        self,
        texts: List,
        type_logs: str,
        d_loaded_logs: Dict
    ) -> Tuple[List[Dict], List[str]]:
        logs, log_hashs = [], []

        for text in texts:
            log_hash = text2hash(text)
            if log_hash not in d_loaded_logs:
                log = {'type': type_logs, 'text': text, 'self': dict(), 'asked': dict()}
                if not (self.use_cache and log_hash in self.hash_files and text != ""):
                    temp=1
                if self.use_cache and log_hash in self.hash_files and text != "":
                    cached_path = os.path.join(self.log_dir, log_hash)
                    try:
                        with open(cached_path, 'r') as f_log:
                            tmp  = json.load(f_log)
                            assert all([k in log for k in ['type', 'text', 'self', 'asked']])
                            assert isinstance(log['type'], str)
                            assert isinstance(log['text'], str)
                            assert isinstance(log['self'], dict)
                            assert isinstance(log['asked'], dict)
                            log = tmp
                    except json.decoder.JSONDecodeError:
                        self.hash_files.remove(log_hash)
                        os.remove(cached_path)
                    except AssertionError:
                        self.hash_files.remove(log_hash)
                        os.remove(cached_path)

                d_loaded_logs[log_hash] = log

            logs.append(d_loaded_logs[log_hash])
            log_hashs.append(log_hash)

        return logs, log_hashs

    def _serialize_logs(
        self,
        logs: List[Dict],
        hashes: List[str]
    ) -> None:
        for log, hash in zip(logs, hashes):
            with open(os.path.join(self.log_dir, hash), 'w') as outfile:
                json.dump(log, outfile, indent=2)

    def open_log_from_text(self, text: str) -> Dict:
        """
        Function to open a serialised log and analyse it.
        """
        log_hash = text2hash(text)
        with open(os.path.join(self.log_dir, log_hash), 'r') as f_log:
            log = json.load(f_log)
        return log

    def _compute_answer_selection(
        self,
        logs: List[Dict],
        type_logs: str
    ) -> None:
        for answer_type in self._get_answer_types(type_logs):
            to_do_exs, to_do_exs_idxs = [], []
            for idx, log in enumerate(logs):
                if answer_type not in log['self'] and log['text'] != '':
                    log['self'][answer_type] = dict()
                    to_do_exs.append(log['text'])
                    to_do_exs_idxs.append(idx)

            if len(to_do_exs) != 0:
                list_answers = self._predict_self_answers(to_do_exs, answer_type)
                for i in range(len(list_answers)):
                    logs[to_do_exs_idxs[i]]['self'][answer_type]['answers'] = list_answers[i]

        return len(to_do_exs) != 0

    def _compute_question_generation(
        self,
        logs: List[Dict],
        type_logs: str
    ) -> None:
        name_model_qg = self._get_qg_hash(type_logs)

        to_do_exs, to_do_exs_idxs, to_do_exs_types = [], [], []
        for idx, log in enumerate(logs):
            if log['text'] == '':
                continue
            for answer_type in self._get_answer_types(type_logs):
                if name_model_qg not in log['self'][answer_type]:
                    log['self'][answer_type][name_model_qg] = {'questions': []}

                    to_do_exs += [(a, log['text']) for a in log['self'][answer_type]['answers']]
                    to_do_exs_idxs += [idx] * len(log['self'][answer_type]['answers'])
                    to_do_exs_types += [answer_type] * len(log['self'][answer_type]['answers'])

        if len(to_do_exs) != 0:
            question_texts = self._predict_questions(to_do_exs, type_logs)
            for i in range(len(question_texts)):
                idx = to_do_exs_idxs[i]
                answer_type = to_do_exs_types[i]
                question = question_texts[i]
                logs[idx]['self'][answer_type][name_model_qg]['questions'].append(question)

        return len(to_do_exs) != 0

    def _compute_question_answering(
        self,
        logs_1: Dict,
        logs_2: Dict,
        type_logs_1: str,
        type_logs_2: str
    ) -> None:
        """
        asking questions from logs_2 on text from logs_1
        """
        assert len(logs_1) == len(logs_2)

        name_model_qg = self._get_qg_hash(type_logs_2)
        name_model_qa = self._get_qa_hash(type_logs_1)

        to_do_exs, to_do_exs_types, to_do_exs_idxs, to_do_gold_asws = [], [], [], []
        for idx, (log_1, log_2) in enumerate(zip(logs_1, logs_2)):
            if log_1['text'] == '' or log_2['text'] == '':
                continue
            for answer_type in self._get_answer_types(type_logs_2):
                questions = log_2['self'][answer_type][name_model_qg]['questions']
                gold_answers = log_2['self'][answer_type]['answers']
                assert len(questions) == len(gold_answers)
                for question, gold_answer in zip(questions, gold_answers):
                    if question not in log_1['asked']:
                        log_1['asked'][question] = dict()

                    if name_model_qa not in log_1['asked'][question]:
                        to_do_exs += [(question, log_1['text'])]
                        to_do_exs_idxs += [idx]
                        to_do_gold_asws += [gold_answer]

                    # if already in the logs, we need to add the gold_answers if it hasnt been yet
                    elif gold_answer not in log_1['asked'][question][name_model_qa]['ground_truth']:
                        log_1['asked'][question][name_model_qa]['ground_truth'][gold_answer] = {}

        if len(to_do_exs) != 0:
            answerability_scores, qa_texts = self._predict_answers(to_do_exs, type_logs_1)

            assert len(to_do_exs) == len(qa_texts) == len(to_do_gold_asws) == len(answerability_scores)
            for i in range(len(to_do_exs)):

                question = to_do_exs[i][0]
                idx = to_do_exs_idxs[i]
                assert to_do_exs[i][1] == logs_1[idx]['text']

                if name_model_qa not in logs_1[idx]['asked'][question]:
                    logs_1[idx]['asked'][question][name_model_qa] = {'answer': qa_texts[i],
                                                                     'answerability': answerability_scores[i],
                                                                     'ground_truth': dict()
                                                                     }
                logs_1[idx]['asked'][question][name_model_qa]['ground_truth'][to_do_gold_asws[i]] = {}

        return len(to_do_exs) != 0

    def _compute_answer_similarity_scores(
        self,
        logs: Dict,
        type_logs: str
    ) -> None:
        """
        filling the similarity scores
        """

        modified_logs = False
        name_model_qa = self._get_qa_hash(type_logs)

        for type_score in self.list_scores:

            # no need for comparison for answerabiliy, it is calculated directly in compute_question_answering
            if type_score == 'answerability':
                continue

            to_do_exs_idxs, to_do_questions, to_do_pred_asws, to_do_gold_asws = [], [], [], []
            for idx, log in enumerate(logs):
                if log['text'] == '':
                    continue
                for question in log['asked']:
                    d_answer = log['asked'][question][self._get_qa_hash(log['type'])]
                    for gold_answer in d_answer['ground_truth']:
                        if type_score not in d_answer['ground_truth'][gold_answer]:
                            to_do_exs_idxs += [idx]
                            to_do_questions += [question]
                            to_do_pred_asws += [d_answer['answer']]
                            to_do_gold_asws += [gold_answer]

            if len(to_do_exs_idxs) != 0:

                modified_logs = True

                if type_score == 'f1':
                    sim_scores = [calculate_f1_squad(pred_asw, gold_asw) for pred_asw, gold_asw in
                                  zip(to_do_pred_asws, to_do_gold_asws)]
                elif type_score == 'bertscore':
                    sim_scores = calculate_BERTScore(to_do_pred_asws, to_do_gold_asws, self.metric_BERTScore,
                                                     device=self.device)
                else:
                    raise NotImplementedError(f"{type_score} not implemented")

                assert len(to_do_exs_idxs) == len(sim_scores)
                for i in range(len(to_do_exs_idxs)):
                    idx = to_do_exs_idxs[i]
                    q = to_do_questions[i]
                    a = to_do_gold_asws[i]
                    logs[idx]['asked'][q][name_model_qa]['ground_truth'][a][type_score] = sim_scores[i]

        return modified_logs

    def _compute_weighter(
        self,
        logs: Dict,
        type_logs: str
    ) -> None:
        """
        weighting the probability that a question is asking about important content or not (see https://arxiv.org/abs/2103.12693)
        """

        name_model_weighter = self._get_weighter_hash()
        name_model_qg = self._get_qg_hash(type_logs)

        to_do_exs, to_do_exs_types, to_do_exs_idxs, to_do_gold_asws = [], [], [], []
        for idx, log in enumerate(logs):
            if log['text'] == '':
                continue
            for answer_type in self._get_answer_types(type_logs):
                if name_model_weighter not in log['self'][answer_type][name_model_qg]:
                    log['self'][answer_type][name_model_qg][name_model_weighter] = []

                    questions = log['self'][answer_type][name_model_qg]['questions']
                    answers = log['self'][answer_type]['answers']
                    assert len(questions) == len(answers)
                    to_do_exs += [f"{asw} {self.sep} {question} {self.sep} {log['text']}"
                                  for asw, question in zip(answers, questions)]

                    to_do_exs_idxs += [idx] * len(answers)
                    to_do_exs_types += [answer_type] * len(answers)

        if len(to_do_exs) != 0:
            weighter_scores = self._predict_weighter(to_do_exs)
            assert len(to_do_exs) == len(weighter_scores)
            for i in range(len(to_do_exs)):
                idx = to_do_exs_idxs[i]
                answer_type = to_do_exs_types[i]
                logs[idx]['self'][answer_type][name_model_qg][name_model_weighter].append(weighter_scores[i])

        return len(to_do_exs) != 0

    def _get_answer_types(self, type_logs: str) -> str:
        return ('TABLE', ) if type_logs == 'src' and self.task == 'data2text' else self.answer_types

    def _predict_self_answers(
        self,
        texts: List,
        answer_type: str
    ) -> List[str]:
        if self.limit_sent is not None:
            list_sentences = [sentencize(text, self.spacy_pipeline) for text in texts]
            texts = [' '.join(sentences[:self.limit_sent]) for sentences in list_sentences]

        list_answers = []
        if answer_type == 'NER':
            list_answers = [[a.text for a in self.spacy_pipeline(text).ents] for text in texts]
        elif answer_type == 'NOUN':
            list_answers = [[a.text for a in self.spacy_pipeline(text).noun_chunks] for text in texts]
        elif answer_type == 'SPANER':
            pass  # todo not implemented
        elif answer_type == 'TABLE':
            list_answers = [extract_table_answers(text) for text in texts]

        return list_answers

    def _predict_questions(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> List[str]:
        model_QG = self.models[type_logs]['QG']

        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        _, question_texts = model_QG.predict(formated_inputs)

        return question_texts

    def _predict_answers(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> Tuple[List[float], List[str]]:
        model_QA = self.models[type_logs]['QA']
        formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        qa_scores, qa_texts = model_QA.predict(formated_inputs)

        return qa_scores, qa_texts

    def _predict_weighter(self, to_do_exs: List[str]) -> List[float]:
        if self.models['Weighter'] is None:
            # Neutral Policy
            probs = [1.0 for _ in to_do_exs]

        else:
            probs, texts = self.models['Weighter'].predict(to_do_exs)
            assert len(probs) == len(to_do_exs)

        return probs

    def _calculate_score_from_logs(
        self,
        hyp_log: List[Dict],
        compared_logs: List[List[Dict]]
    ) -> float:

        scores = []
        for compared_log in compared_logs:
            if compared_log['text'] == '' or hyp_log['text'] == '':
                score = 0
            else:
                hyp_score = self._base_score(hyp_log, compared_log)
                compared_score = self._base_score(compared_log, hyp_log)
                score = np.average([hyp_score, compared_score])
            scores.append(score)
        return self.reduction_multi_refs(scores)

    def _base_score(
        self,
        questioned_log: Dict,
        compared_log: Dict
    ) -> float:
        regularizer = lambda list_score, list_reg: np.multiply(scores, list_reg).tolist()
        list_borned = lambda a_list: [max(min(1, x), 0) for x in a_list]

        if self.do_consistency:
            consistencies = self._get_scores(compared_log, compared_log, 'f1')

        if self.do_weighter and compared_log['type'] == 'src':
            name_model_qg = self._get_qg_hash(compared_log['type'])
            name_model_weighter = self._get_weighter_hash()
            weighter_probs = [
                w for answer_type in self._get_answer_types(questioned_log['type'])
                for w in compared_log['self'][answer_type][name_model_qg][name_model_weighter]
            ]

        list_scores = []
        for type_score in self.list_scores:
            scores = self._get_scores(questioned_log, compared_log, type_score)

            # if no questions, return a score set to 0; could be improved though ?
            if len(scores) == 0:
                return 0

            # sometimes the answers scores return a value ~1.000000X which is superior to 1
            scores = list_borned(scores)

            if self.do_consistency:
                assert consistencies is not None, "consistencies is None. Please compute the score with ques_consists activate."
                scores = regularizer(scores, consistencies)

            if self.do_weighter and compared_log['type'] == 'src':
                assert weighter_probs is not None, "weighter_probs is None. Please compute the weighter probs with do_weighter activate."
                scores = regularizer(scores, weighter_probs)

            list_scores += scores

        final_score = np.average(list_scores)
        assert 0 <= final_score <= 1, "score should be in [0-1] "
        return final_score

    def _get_scores(
        self,
        questioned_log: List[Dict],
        compared_log: List[Dict],
        type_score: str
    ) -> List[float]:

        name_model_qg = self._get_qg_hash(compared_log['type'])
        asked_questions = [q for answer_type in self._get_answer_types(compared_log['type'])
                           for q in compared_log['self'][answer_type][name_model_qg]['questions']
                           ]

        name_model_qa = self._get_qa_hash(questioned_log['type'])
        if type_score == 'answerability':
            scores = [questioned_log['asked'][q][name_model_qa]['answerability']
                      for q in asked_questions]

        else:  # F1 or BERTScore
            asked_answers = [a for answer_type in self._get_answer_types(compared_log['type'])
                             for a in compared_log['self'][answer_type]['answers']]

            assert len(asked_answers) == len(asked_questions)

            [questioned_log['asked'][q][name_model_qa]['ground_truth'][a][type_score]
             for q, a in zip(asked_questions, asked_answers)]
            scores = [questioned_log['asked'][q][name_model_qa]['ground_truth'][a][type_score]
                      for q, a in zip(asked_questions, asked_answers)]

        return scores

    def get_model(self, model_name: str,):
        keep_score_idx = None

        if 't5' in model_name.lower():

            if "qa" in model_name.lower():
                # 73 is the index for the token unanswerable in T5 vocabulary
                keep_score_idx = 73
            if 'weighter' in model_name.lower():
                # 1176 is the index for the token true in T5 vocabulary
                keep_score_idx = 1176
            if model_name == f"{HF_ORGANIZATION}/t5-qg_squad1-en":
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'

            # batch size
            model_batch_size = self.qg_batch_size if "qg" in model_name.lower() else self.clf_batch_size

            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device
            )

        else:
            raise NotImplementedError(f'Model Name Not Handled: the model name should contain t5 ({model_name}).')

        return model

    def set_model(
        self,
        key: str,
        task: str,
        model_name: str,
    ) -> None:

        assert key in [None, 'hyp', 'src', 'ref']
        assert task in ['weighter', 'QG', 'QG']

        model = self.get_model(model_name=model_name)

        if key is None:
            self.models[task] = model
        else:
            self.models[key][task] = model

    def _get_answer_hash(self) -> str:
        # TODO: self.spacy_pipeline
        msg = f"LimitSent={self.limit_sent}" \
              f"_models={'_'.join(self.answer_types)}"

        return msg

    def _get_qg_hash(self, type_log: str) -> str:
        model = self.models[type_log]['QG']
        msg = f'QG_hash={model.pretrained_model_name_or_path}'

        return msg

    def _get_qa_hash(self, type_log: str) -> str:
        model = self.models[type_log]['QA']
        msg = f'QA_hash={model.pretrained_model_name_or_path}'

        return msg

    def _get_weighter_hash(self) -> str:
        msg = 'W_hash='
        tmp = 'None'
        if self.do_weighter:
            model = self.models['Weighter']
            tmp = f'{model.pretrained_model_name_or_path}'
        msg += tmp
        return msg

    def __hash__(self) -> str:
        msg = f"QuestEval_version={__version__}" \
              f"_task={self.task}_lang={self.language}_preproc={self.src_preproc_pipe}" \
              f"_consist={self.do_consistency}_scores={self.list_scores}" \
              f"{self._get_weighter_hash()}" \
              f"_hyp_{self._get_qa_hash('hyp')}_ref_{self._get_qa_hash('ref')}_src_{self._get_qa_hash('src')}" \
              f"_hyp_{self._get_qg_hash('hyp')}_ref_{self._get_qg_hash('ref')}_src_{self._get_qg_hash('src')}"

        return msg
