from typing import List, Tuple
import os
import json
import urllib
import hashlib # todo setup.py
import zipfile
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
import transformers
from questeval import DIR, _version_
from questeval.utils import (
    APIQA,
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers
)
#from questeval.uni_utils import T2tUniModel todo

ZIPPED_MODELS_URL = "https://safeval.s3.eu-west-3.amazonaws.com"

# todo : refact model loader wrt PAD

# todo
#  train weighter / remove prefixe
#  train QG sur squad / QA synth
#  remove prefixe


class QuestEval:
    def __init__(
        self,
        task: str = "text2text",
        language: str = "en",
        answer_types: Tuple = ('NER', 'NOUN'),
        list_scores: Tuple = ('bertscore', 'answerability', 'f1'),
        src_preproc_pipe=None,
        do_weighter: bool = False,
        do_consistency: bool = False,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        QG_top_p: float = 1.0,
        isCuda: bool = True
    ):
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


        self.AVAILABLE_LANGUAGES = ("en", "multi")
        self.AVAILABLE_TASKS = ("text2text", "summarization", "text_simplification", "data2text")

        if task not in self.AVAILABLE_TASKS:
            print(f"Task {task} is not known. Setting the default text2text task. ")
            task="text2text"

        if language not in self.AVAILABLE_LANGUAGES:
            print(f"Language {language} is not known. Setting the default multilingual models.")
            language = 'multi'

        if task == 'summarization' and do_weighter == False:
            logging.warning("Task is summarization but the weighter is deactivate. Set do_weighter=True to activate it when loading QuestEval.")

        self.log_dir = os.path.join(DIR, 'logs')
        self.hash_files = set(os.listdir(self.log_dir))

        self.task = task
        self.language = language

        self.answer_types = answer_types

        self.src_preproc_pipe = src_preproc_pipe
        self.limit_sent = limit_sent
        self.sep = "</s>"
        self.qg_prefix = None

        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.QG_top_p = QG_top_p
        self.device = 'cuda' if (torch.cuda.is_available() and isCuda) else 'cpu'

        self.reduction_multi_refs = max
        self.do_consistency = do_consistency
        self.do_weighter = do_weighter
        self.list_scores = list_scores
        if 'bertscore' in self.list_scores:
            self.metric_BERTScore = load_metric("bertscore")

        self.keep_score_idx = None # will be defined later in load_model wrt to the QA model loaded
        self.models = self.load_all_models()

        if self.src_preproc_pipe == None:
            """
            structured tables should be linearized for our QA/QG format this way:
            'name [ The Eagle ] , eatType [ coffee shop ] , food [ French ] , priceRange [ £ 2 0 - 2 5 ] , customer rating [ 3 out of 5 ] , area [ city centre ] , familyFriendly [ yes ] , near [ Burger King ]'
            we handle by default the preprocessing of the table for webnlg given the GEM format (https://gem-benchmark.com/)
            if your tables are in an other format, please pass a custom function for src_preproc_pipe
            """
            if task == 'data2text':
                from questeval.data2text_objects.data_formating import linearize_webnlg_input
                self.src_preproc_pipe = linearize_webnlg_input

        if language == 'en':
            self.spacy_pipeline = spacy.load('en_core_web_sm')
        else:
            logging.warning("Removing 'NOUN CHUNK' from the candidate answers in answer_types. It is not yet handled for non English texts.")
            self.answer_types = ('NER')
            self.spacy_pipeline = spacy.load('xx_ent_wiki_sm')


    def corpus_questeval(self,
                         hypothesis: List[str],
                         sources: List[str] = None,
                         list_references: List[List[str]] = None,
                         batch_size: int = 512):

        assert sources is not None or list_references is not None, "you need to provide at least the sources or the references"
        if list_references is not None:
            assert len(list_references) == len(hypothesis)
        if sources is not None:
            assert len(sources) == len(hypothesis)


        scores = []
        nb_batch = len(range(0, len(hypothesis), batch_size))
        for batch_idx in range(0, len(hypothesis), batch_size):
            logging.info(f"Proceeding batch {batch_idx}/{nb_batch}")
            batch_sources, batch_list_references = None, None
            if sources is not None:
                batch_sources = sources[batch_idx:batch_idx+batch_size]
            if list_references is not None:
                batch_list_references = list_references[batch_idx:batch_idx+batch_size]
            scores += self._batch_questeval(hypothesis=hypothesis[batch_idx:batch_idx+batch_size],
                                           sources=batch_sources,
                                           list_references=batch_list_references,
            )

        result = {'corpus_score': np.average(scores), 'ex_level_scores': scores}
        return result


    def _batch_questeval(self,
                        hypothesis: List[str],
                        sources: List[str] = None,
                        list_references: List[List[str]] = None,
    ):

        list_compared_logs = []

        # Hypothesis
        hyp_logs, hyp_hashes = self.texts2logs(hypothesis, type_logs='hyp')

        # Source
        if sources is not None:
            src_logs, src_hashes = self.texts2logs(sources, type_logs='src')
            # Asking the questions on the compared text
            self.compute_question_answering(src_logs, hyp_logs, 'src', 'hyp')
            self.compute_question_answering(hyp_logs, src_logs, 'hyp', 'src')
            # Compute the similarity scores
            self.compute_answer_similarity_scores(src_logs)
            # Serialise logs
            self.serialize_logs(src_logs, src_hashes)
            list_compared_logs.append(src_logs)

        # Reference
        if list_references is not None:
            for references in list_references:
                ref_logs, ref_hashes = self.texts2logs(references, type_logs='ref')
                # Asking the questions on the compared text
                self.compute_question_answering(ref_logs, hyp_logs, 'ref', 'hyp')
                self.compute_question_answering(hyp_logs, ref_logs, 'hyp', 'ref')
                # Compute the similarity scores
                self.compute_answer_similarity_scores(ref_logs)
                # Serialise logs
                self.serialize_logs(ref_logs, ref_hashes)
                list_compared_logs.append(ref_logs)

        # Compute the similarity scores for hyp
        self.compute_answer_similarity_scores(hyp_logs)
        # Serialise hyp logs
        self.serialize_logs(hyp_logs, hyp_hashes)
        list_compared_logs = [[list_compared_logs[i][j]
                              for i in range(len(list_compared_logs))]
                              for j in range(len(list_compared_logs[0]))
                              ]

        # Calculate Score
        scores = []
        for hyps_log, compared_logs in zip(hyp_logs, list_compared_logs):
            scores.append(self.calculate_score_from_logs(hyps_log, compared_logs))

        return scores


    @staticmethod
    def get_log_hash(string: str
    ) -> str:
        hash_object = hashlib.sha512(string.encode('utf-8'))
        hex_dig = hash_object.hexdigest()
        return hex_dig


    def load_logs(self,
                  texts: list,
                  type_logs: str
    ):

        logs, log_hashs = [], []
        for text in texts:
            log_hash = self.get_log_hash(text)
            if log_hash in self.hash_files:
                with open(os.path.join(self.log_dir, log_hash), 'r') as f_log:
                    logs.append(json.load(f_log))
                    log_hashs.append(log_hash)
            else:
                logs.append({'type': type_logs, 'text': text, 'self': dict(), 'asked': dict()})
                log_hashs.append(log_hash)
        return logs, log_hashs


    def serialize_logs(self,
                       logs: List[dict],
                       hashes: List[str]
    ):
        for log, hash in zip(logs, hashes):
            with open(os.path.join(self.log_dir, hash), 'w') as outfile:
                json.dump(log, outfile)


    def texts2logs(self,
                   texts: List[str],
                   type_logs: str
    ):

        # Preprocessing
        if type_logs == 'src' and self.src_preproc_pipe is not None:
            texts = [self.src_preproc_pipe(source) for source in texts]

        logs, logs_hashes = self.load_logs(texts, type_logs)
        # Selecting the answers
        self.compute_answer_selection(logs, type_logs)
        #  Generating the questions
        self.compute_question_generation(logs, type_logs)
        # Asking the questions on itself (Round trip consistency)
        if self.do_consistency:
            self.compute_question_answering(logs, logs, type_logs, type_logs)
        # Weighter
        if type_logs == 'src' and self.do_weighter:
            self.compute_weighter(logs, type_logs='src')

        return logs, logs_hashes


    def compute_answer_selection(self,
                                 logs: List[str],
                                 type_logs: str
    ):

        for answer_type in self.get_answer_types(type_logs):
            to_do_exs, to_do_exs_idxs = [], []
            for idx, log in enumerate(logs):
                if answer_type not in log['self']:
                    log['self'][answer_type] = dict()
                    to_do_exs.append(log['text'])
                    to_do_exs_idxs.append(idx)


            if len(to_do_exs) != 0:
                list_answers = self.predict_self_answers(to_do_exs, answer_type)
                for i in range(len(list_answers)):
                    logs[to_do_exs_idxs[i]]['self'][answer_type]['answers'] = list_answers[i]



    def compute_question_generation(self,
                                    logs: list,
                                    type_logs: str
    ):

        name_qg_model = "temp" # todo

        to_do_exs, to_do_exs_idxs, to_do_exs_types = [], [], []
        for idx, log in enumerate(logs):
            for answer_type in self.get_answer_types(type_logs):
                if name_qg_model not in log['self'][answer_type]:
                    log['self'][answer_type][name_qg_model] = {'questions':[]}

                    to_do_exs += [(a, log['text']) for a in log['self'][answer_type]['answers']]
                    to_do_exs_idxs += [idx] * len(log['self'][answer_type]['answers'])
                    to_do_exs_types += [answer_type] * len(log['self'][answer_type]['answers'])

        if len(to_do_exs) != 0:
            question_texts = self.predict_questions(to_do_exs, type_logs)
            for i in range(len(question_texts)):

                idx = to_do_exs_idxs[i]
                answer_type = to_do_exs_types[i]
                question = question_texts[i]
                logs[idx]['self'][answer_type][name_qg_model]['questions'].append(question)


    def compute_question_answering(self,
                                   logs_1: dict,
                                   logs_2: dict,
                                   type_logs_1: str,
                                   type_logs_2: str
    ):
        """
        asking questions from logs_2 on text from logs_1
        """
        assert len(logs_1) == len(logs_2)

        name_model_QG = 'temp' # todo
        name_model_QA = 'temp' # todo

        to_do_exs, to_do_exs_types, to_do_exs_idxs, to_do_gold_asws = [], [], [], []
        for idx, (log_1, log_2) in enumerate(zip(logs_1, logs_2)):
            for answer_type in self.get_answer_types(type_logs_2):
                questions = log_2['self'][answer_type][name_model_QG]['questions']
                gold_answers = log_2['self'][answer_type]['answers']
                assert len(questions) == len(gold_answers)
                for question, gold_answer in zip(questions, gold_answers):
                    if question not in log_1['asked']:
                        log_1['asked'][question] = dict()

                    if name_model_QA not in log_1['asked'][question]:
                        to_do_exs += [(question, log_1['text'])]
                        to_do_exs_idxs += [idx]
                        to_do_gold_asws += [gold_answer]

                    # if already in the logs, we need to add the gold_answers if it hasnt been yet
                    elif gold_answer not in log_1['asked'][question][name_model_QA]['ground_truth']:
                          log_1['asked'][question][name_model_QA]['ground_truth'][gold_answer] = {}

        if len(to_do_exs) != 0:
            qa_scores, qa_texts = self.predict_answers(to_do_exs, type_logs_1)

            answerability_scores = [1 - qa_score[0] for qa_score in qa_scores[self.keep_score_idx]] # todo qa_score[0] ?

            assert len(to_do_exs) == len(qa_texts) == len(to_do_gold_asws) == len(answerability_scores)
            for i in range(len(to_do_exs)):

                question = to_do_exs[i][0]
                idx = to_do_exs_idxs[i]
                assert to_do_exs[i][1] == logs_1[idx]['text']

                if name_model_QA not in logs_1[idx]['asked'][question]:
                    logs_1[idx]['asked'][question][name_model_QA] = {'answer': qa_texts[i],
                                                                     'answerability': answerability_scores[i],
                                                                     'ground_truth': dict()
                                                                     }
                logs_1[idx]['asked'][question][name_model_QA]['ground_truth'][to_do_gold_asws[i]] = {}


    def compute_answer_similarity_scores(self,
                                         logs: dict,
    ):
        """
        filling the similarity scores
        """

        name_model_QA = 'temp'  # todo

        for type_score in self.list_scores:

            # no need for comparison for answerabiliy, it is calculated directly in compute_question_answering
            if type_score == 'answerability':
                continue

            to_do_exs_idxs, to_do_questions, to_do_pred_asws, to_do_gold_asws = [], [], [], []
            for idx, log in enumerate(logs):
                for question in log['asked']:
                    d_answer = log['asked'][question][name_model_QA]
                    for gold_answer in d_answer['ground_truth']:

                        if type_score not in d_answer['ground_truth'][gold_answer]:

                            to_do_exs_idxs += [idx]
                            to_do_questions += [question]
                            to_do_pred_asws += [d_answer['answer']]
                            to_do_gold_asws += [gold_answer]

            if len(to_do_exs_idxs) != 0:
                if type_score == 'f1':
                    sim_scores = [calculate_f1_squad(pred_asw, gold_asw) for pred_asw, gold_asw in zip(to_do_pred_asws, to_do_gold_asws)]
                elif type_score == 'bertscore':
                    sim_scores = calculate_BERTScore(to_do_pred_asws, to_do_gold_asws, self.metric_BERTScore, device=self.device)
                else:
                    raise NotImplementedError(f"{type_score} not implemented")

                assert len(to_do_exs_idxs) == len(sim_scores)
                for i in range(len(to_do_exs_idxs)):
                    idx = to_do_exs_idxs[i]
                    q = to_do_questions[i]
                    a = to_do_gold_asws[i]
                    logs[idx]['asked'][q][name_model_QA]['ground_truth'][a][type_score] = sim_scores[i]


    def compute_weighter(self,
                         logs: dict,
                         type_logs:str
    ):
        """
        weighting the probability that a question is asking about important content or not (see https://arxiv.org/abs/2103.12693)
        """

        name_model_weighter = 'temp' # todo
        name_model_QG = 'temp'

        to_do_exs, to_do_exs_types, to_do_exs_idxs, to_do_gold_asws = [], [], [], []
        for idx, log in enumerate(logs):
            for answer_type in self.get_answer_types(type_logs):
                if name_model_weighter not in log['self'][answer_type][name_model_QG]:
                    log['self'][answer_type][name_model_QG][name_model_weighter] = []

                    questions = log['self'][answer_type][name_model_QG]['questions']
                    answers = log['self'][answer_type]['answers']
                    assert len(questions) == len(answers)
                    to_do_exs += [f"{asw} {self.sep} {question} {self.sep} {log['text']}"
                                  for asw, question in zip(answers, questions)]

                    to_do_exs_idxs += [idx] * len(answers)
                    to_do_exs_types += [answer_type] * len(answers)

        if len(to_do_exs) != 0:
            weighter_scores = self.predict_weighter(to_do_exs)
            assert len(to_do_exs) == len(weighter_scores)
            for i in range(len(to_do_exs)):
                idx = to_do_exs_idxs[i]
                answer_type = to_do_exs_types[i]
                logs[idx]['self'][answer_type][name_model_QG][name_model_weighter].append(weighter_scores[i])


    def get_answer_types(self,
                         type_logs: str
    ) -> str:
        return 'TABLE' if type_logs == 'src' and self.task == 'data2text' else self.answer_types


    def predict_self_answers(self,
                             texts: list,
                             answer_type: str
    ) -> List[str]:

        if self.limit_sent is not None:
            list_sentences = [sentencize(text, self.spacy_pipeline) for text in texts]
            texts = [' '.join(sentences[:self.limit_sent]) for sentences in list_sentences]

        if answer_type == 'NER':
            list_answers = [[a.text for a in self.spacy_pipeline(text).ents] for text in texts]
        elif answer_type == 'NOUN':
            list_answers = [[a.text for a in self.spacy_pipeline(text).noun_chunks] for text in texts]
        elif answer_type == 'SPANER':
            pass  # todo not implemented
        elif answer_type == 'TABLE':
            list_answers = [extract_table_answers(text) for text in texts]
        return list_answers


    def predict_questions(self,
                          to_do_exs: List[tuple],
                          type_logs: str
    ) -> List[str]:

        model_QG = self.models[type_logs]['QG']

        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        question_probs, question_texts = model_QG.predict(formated_inputs, beam_size=1)
        return question_texts


    def predict_answers(self,
                        to_do_exs: List[tuple],
                        type_logs: str
    ):

        model_QA = self.models[type_logs]['QA']
        formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        qa_scores, qa_texts = model_QA.predict(formated_inputs)
        return qa_scores, qa_texts


    def predict_weighter(self,
                         to_do_exs
    ) -> List[float]:
        # Neutral Policy
        if self.models['Weighter'] is None:
            return [1.0 for _ in to_do_exs]
        else:
            probs, texts = self.models['Weighter'].predict(to_do_exs)
            if len(to_do_exs) == 1:
                probs['max'] = [probs['max']]

            # when used as a classifier, only the first token prob is enough to evaluate the probability.
            probs = [prob[0] if text == "true" else 1 - prob[0]
                                for text, prob in zip(texts, probs['max'])]
            probs = [prob.item() for prob in probs]
            assert len(probs) == len(to_do_exs)
            return probs


    def calculate_score_from_logs(self,
                                  hyp_log:List[dict],
                                  compared_logs:List[List[dict]]
    ) -> float:
        scores = []
        for compared_log in compared_logs:
            hyp_score = self.base_score(hyp_log, compared_log)
            compared_score = self.base_score(compared_log, hyp_log)
            scores.append(np.average([hyp_score, compared_score]))
        return self.reduction_multi_refs(scores)


    def base_score(self,
                   questioned_log: dict,
                   compared_log: dict
    ) -> float:

        regularizer = lambda list_score, list_reg: np.multiply(scores, list_reg).tolist()
        list_borned = lambda a_list: [max(min(1, x), 0) for x in a_list]

        name_model_qg = 'temp' # todo
        name_model_weighter = 'temp' # todo

        if self.do_consistency:
            consistencies = self.get_scores(compared_log, compared_log, 'f1')

        if self.do_weighter and questioned_log['type'] == 'src':
            weighter_probs = [w for answer_type in self.get_answer_types(compared_log['type'])
                              for w in questioned_log['self'][answer_type][name_model_qg][name_model_weighter]
            ]

        list_scores = []
        for type_score in self.list_scores:
            scores = self.get_scores(questioned_log, compared_log, type_score)

            # if no questions, return a score set to 0; could be improved though ?
            if len(scores) == 0:
                return 0

            # sometimes the answers scores return a value ~1.000000X which is superior to 1
            scores = list_borned(scores)

            if self.do_consistency:
                assert consistencies is not None, "consistencies is None. Please compute the score with ques_consists activate."
                scores = regularizer(scores, consistencies)

            if self.do_weighter:
                assert weighter_probs is not None, "weighter_probs is None. Please compute the weighter probs with do_weighter activate."
                scores = regularizer(scores, weighter_probs)

            list_scores += scores

        final_score = np.average(list_scores)
        assert 0 <= final_score <= 1, "score should be in [0-1] "
        return final_score


    def get_scores(self,
                   questioned_log: List[dict],
                   compared_log: List[dict],
                   type_score: str
    ) -> List[float]:

        name_model_qa = 'temp'
        name_model_qg = 'temp'

        asked_questions = [q for answer_type in self.get_answer_types(compared_log['type'])
                           for q in compared_log['self'][answer_type][name_model_qg]['questions']
                           ]

        if type_score == 'answerability':
            scores =  [questioned_log['asked'][q][name_model_qa]['answerability']
                       for q in asked_questions]

        else: # F1 or BERTScore
            asked_answers = [a for answer_type in self.get_answer_types(compared_log['type'])
                             for a in compared_log['self'][answer_type]['answers']]

            assert len(asked_answers) == len(asked_questions)

            try:
                [questioned_log['asked'][q][name_model_qa]['ground_truth'][a][type_score]
                 for q, a in zip(asked_questions, asked_answers)]
            except:
                t=0
            scores = [questioned_log['asked'][q][name_model_qa]['ground_truth'][a][type_score]
                      for q, a in zip(asked_questions, asked_answers)]

        return scores


    def load_all_models(self
    ):
        # Textual hypothesis
        models = {"hyp":{}}
        if self.language == 'en':
            models['hyp']['QA'] = f"models/QA_en_T5"
            models['hyp']['QG'] = f"models/QG_en_T5"
        else:
            models['hyp']['QA'] = f"models/QA_multilingual_question_in_english_minilm"
            models['hyp']['QG'] = f"models/QG_multilingual_question_in_english_minilm"

        # (if) multimodal sources
        if self.task == "data2text":
            models['src'] = dict()
            models['src']['QA'] = f"models/QA_webnlg_T5"
            models['src']['QG'] = f"models/QG_webnlg_T5"

        # Loading all the different models
        for modality in models.keys():
            for task in models[modality].keys():
                if not type(models[modality][task]) == str:
                    continue
                models[modality][task] = self.generic_load_model(models[modality][task], is_task_QG=task=='QG')

        # Loading the weighter
        models['Weighter'] = None
        if self.do_weighter:
            models['Weighter'] = self.generic_load_model(f"models/Weighter_en_T5", is_task_QG=False)

        # Linking already loaded models for the other keys
        for k in ["src", "ref"]:
            if models.get(k) == None:
                models[k] = dict()
                models[k]['QA'] = models['hyp']['QA']
                models[k]['QG'] = models['hyp']['QG']

        return models


    def generic_load_model(self,
                           path_model: str,
                           is_task_QG: bool=False
    ):
        # Download the model
        if not os.path.exists(os.path.join(DIR, path_model)):
            if not os.path.exists(os.path.join(DIR, 'models/')):
                os.mkdir(os.path.join(DIR, 'models/'))
            logging.info("Downloading models...")
            zip_model_path = os.path.join(DIR, path_model + '.zip')
            zip_model_url = f"{ZIPPED_MODELS_URL}/{path_model.replace('models/', '')}.zip"
            urllib.request.urlretrieve(zip_model_url, zip_model_path)
            with zipfile.ZipFile(zip_model_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(DIR, 'models/'))

            logging.info("Removing archive...")
            os.remove(zip_model_path)

        # Load the model
        path_model = os.path.join(DIR, path_model)
        assert not ('t5' in path_model.lower() and ('unilm' in path_model.lower() or 'minilm' in path_model.lower()))
        if 't5' in path_model.lower():
            type_model = 't5'
        elif 'unilm' in path_model.lower() or 'minilm' in path_model.lower():
            type_model = 'unilm'
        else:
            raise NotImplementedError(f'Model Name Not Handled: the path should contain t5, unilm or minilm ({path_model}).')

        top_p, keep_score_idx = None, None
        if is_task_QG == "QG":
            model_batch_size = self.qg_batch_size
            top_p = self.QG_top_p
        else:
            # 73 / 220 are for the unanswerable token in T5 / minilm vocabulary
            if type_model == 't5': self.keep_score_idx = 73 # todo move that to init wrt PAD changes
            elif type_model == 'unilm': self.keep_score_idx = 220
            keep_score_idx = [self.keep_score_idx]
            model_batch_size = self.clf_batch_size

        model = self.get_model(path_model=path_model,
                               model_batch_size=model_batch_size,
                               device=self.device,
                               type_model=type_model,
                               top_p=self.QG_top_p,
                               keep_score_idx=keep_score_idx)

        if is_task_QG and self.language == "en":
            # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
            self.qg_prefix = 'sv1' # todo remove that

        return model


    def get_model(self,
                  path_model: str,
                  model_batch_size: str,
                  device: str ='cuda',
                  type_model: str = 't5',
                  keep_score_idx: List[int] = [],
                  top_p: float = None
    ):
        if type_model == 't5':
            load_model = load_t2t_model
        elif type_model == 'unilm':
            load_model = None #load_t2t_uni_model #todo

        model = load_model(path_model, device, keep_score_idx, model_batch_size, top_p)

        return model


    def set_model(self,
                  key: str,
                  task: str,
                  path_model: str,
                  model_batch_size: int = 32,
                  device: str = 'cuda',
                  type_model: str = 't5',
                  keep_score_idx: List[int] = [],
                  top_p: float = None
    ):

        assert key in [None, 'hyp', 'src', 'ref']
        assert task in ['weighter', 'QG', 'QG']

        model = self.get_model(path_model, model_batch_size, device, type_model, keep_score_idx, top_p)

        if key is None:
            self.models[task] = model
        else:
            self.models[task][key] = model


    def get_answer_hash(self
    ) -> str:
        self.spacy_pipeline
        msg = f"LimitSent={self.limit_sent}_models={'_'.join(self.answer_types)}"
        return msg


    def get_qg_hash(self,
                    type_log: str
    ) -> str:
        model = self.models[type_log]['QG']
        msg = f'QG_{type_log}_hash={model.__hash__()}_top_p={model.top_p}'
        return msg


    def get_qa_hash(self,
                    type_log: str
    ) -> str:
        model = self.models[type_log]['QA']
        msg = f'QA_{type_log}_hash={model.__hash__()}'
        return msg


    def get_weighter_hash(self
    ) -> str:
        msg = None
        if self.do_weighter:
            model = self.models['Weighter']
            msg = f'W_hash={model.__hash__()}'
        return msg


    def __hash__(self
    ) -> str:
        msg = f"QuestEval_version={_version_}" \
              f"_task={self.task}_lang={self.language}_preproc={self.src_preproc_pipe}" \
              f"_consist={self.do_consistency}_scores={self.list_scores}" \
              f"_weighter={self.get_weighter_hash()}" \
              f"_{self.get_qa_hash('hyp')}_{self.get_qa_hash('ref')}__{self.get_qa_hash('src')}" \
              f"_{self.get_qg_hash('hyp')}_{self.get_qg_hash('ref')}__{self.get_qg_hash('src')}"

        return msg




def load_t2t_model(path_model,
                   device,
                   keep_score_idx=[],
                   model_batch_size=48,
                   top_p = None
):
    model = APIQA(
        pretrained_model_name_or_path=path_model,
        max_source_length=512,
        model_batch_size=model_batch_size,
        keep_score_idx=keep_score_idx,
        device=device
    )
    if top_p is not None:
        model.top_p = top_p
    return model


"""
def load_t2t_uni_model(
    path_model,
    device,
    keep_score_idx=[],
    model_batch_size=48,
    top_p = None
):
    isCuda = device != 'cpu'
    model = T2tUniModel(
        path_model,
        isCuda,
        keep_score_idx,
        model_batch_size,
    )
    if top_p is not None:
        model.top_p = top_p

    return model
"""