from typing import List, Tuple
import os
import json
import urllib
import zipfile
import numpy as np
import logging
from datasets import load_metric
import spacy
from questeval import DIR
from questeval.utils import (
    APIQA,
    split_on_punct,
    calculate_f1_squad,
)
from questeval.uni_utils import T2tUniModel

ZIPPED_MODELS_URL = "https://safeval.s3.eu-west-3.amazonaws.com"


class QuestEval:
    def __init__(
        self,
        task="text2text",
        language="en",
        limit_sent=5,
        answer_types=['NER', 'NOUN'],
        src_preproc_pipe=None,
        qg_beam_size=1,
        QG_top_p=1,
        lambda_penalty=-0.0,
        do_weighter=False,
        do_BERTScore=False,
        do_regularize=False,
        do_consistency=False,
        qg_batch_size=None,
        clf_batch_size=None,
        isCuda=False
    ):

        self.AVAILABLE_LANGUAGES = ["en", "multi"]
        self.AVAILABLE_TASKS = ["text2text", "summarization", "text_simplification", "E2E", "webnlg"]

        if task not in self.AVAILABLE_TASKS:
            print(f"Task {task} is not known. Setting the default text2text task. ")
            task="text2text"

        if language not in self.AVAILABLE_LANGUAGES:
            print(f"Language {language} is not known. Setting the default multilingual models.")
            language = 'multi'

        self.cache_mode = None
        self.cached_questions_article = None
        self.cache_index = None

        self.task = task
        self.language = language

        self.src_preproc_pipe = src_preproc_pipe
        self.sep = "</s>"
        self.qg_prefix = None
        self.answer_types = answer_types
        self.limit_sent = limit_sent
        self.qg_beam_size = qg_beam_size
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.QG_top_p = QG_top_p
        self.lambda_penalty = lambda_penalty
        self.do_regularize = do_regularize
        self.do_consistency = do_consistency
        self.do_weighter = do_weighter
        self.do_BERTScore = do_BERTScore
        self.metric_BERTScore = load_metric("bertscore")
        self.device = 'cuda' if isCuda else 'cpu'

        self.keep_score_idx = None # will be defined later in load_model wrt to the QA model loaded

        if self.qg_batch_size  is None:
            # Should fit for a 11G ram GPU.
            self.qg_batch_size = max(36 // self.qg_beam_size, 1)
        if self.clf_batch_size is None:
            self.clf_batch_size = 48

        self.models = self.load_all_models()

        if self.src_preproc_pipe == None:
            # table should be linearized for our QA/QG format this way:
            # 'name [ The Eagle ] , eatType [ coffee shop ] , food [ French ] , priceRange [ £ 2 0 - 2 5 ] , customer rating [ 3 out of 5 ] , area [ city centre ] , familyFriendly [ yes ] , near [ Burger King ]'
            # we handle by default the preprocessing of the table for webnlg and e2e given the GEM format (https://gem-benchmark.com/)
            # if your tables are in an other format, please pass a custom function for src_preproc_pipe
            if task == 'E2E':
                from questeval.data2text_objects.data_formating import linearize_e2e_input
                self.src_preproc_pipe = linearize_e2e_input
            elif task == 'webnlg':
                from questeval.data2text_objects.data_formating import linearize_webnlg_input
                self.src_preproc_pipe = linearize_webnlg_input

        if language == 'en':
            self.spacy_pipeline = spacy.load('en_core_web_sm')
        else:
            logging.warning("Removing 'NOUN CHUNK' from the candidate answers in answer_types. It is not yet handled for non English texts.")
            self.answer_types.remove('NOUN')
            self.spacy_pipeline = spacy.load('xx_ent_wiki_sm')


    def generic_load_model(self, path_model, is_task_QG=False):
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
            load_model = load_t2t_model
        elif 'unilm' in path_model.lower() or 'minilm' in path_model.lower():
            load_model = load_t2t_uni_model
        else:
            raise ('Model Name Not Handled.')

        if is_task_QG == "QG":
            model = load_model(path_model, self.device, model_batch_size=self.qg_batch_size)
            model.top_p = self.QG_top_p
            if self.language == "en":
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'
        else:
            # 73 / 220 are for the unanswerable token in T5 / minilm vocabulary
            if 't5' in path_model.lower():
                self.keep_score_idx = 73
            elif 'minilm' in path_model.lower() or 'unilm' in path_model.lower():
                self.keep_score_idx = 220
            model = load_model(path_model, self.device, keep_score_idx=[self.keep_score_idx], model_batch_size=self.clf_batch_size)

        return model

    def load_all_models(self):
        models = {"hyp":{}}
        if self.language == 'en':
            models['hyp']['QA'] = f"models/QA_en_T5"
            models['hyp']['QG'] = f"models/QG_en_T5"
        else:
            models['hyp']['QA'] = f"models/QA_multilingual_question_in_english_minilm"
            models['hyp']['QG'] = f"models/QG_multilingual_question_in_english_minilm"

        if self.task == "E2E":
            models['src'] = dict()
            models['src']['QA'] = f"models/QA_E2E_T5"
            models['src']['QG'] = f"models/QG_E2E_T5"

        if self.task == "webnlg":
            models['src'] = dict()
            models['src']['QA'] = f"models/QA_webnlg_T5"
            models['src']['QG'] = f"models/QG_webnlg_T5"

        for modality in models.keys():
            for task in models[modality].keys():

                if not type(models[modality][task]) == str:
                    continue
                models[modality][task] = self.generic_load_model(models[modality][task], is_task_QG=task=='QG')

        models['Weighter'] = None
        if self.do_weighter:
            models['Weighter'] = self.generic_load_model(f"models/Weighter_en_T5", is_task_QG=False)

        # linking already loaded models for other keys
        for k in ["src", "ref"]:
            if models.get(k) == None:
                models[k] = dict()
                models[k]['QA'] = models['hyp']['QA']
                models[k]['QG'] = models['hyp']['QG']

        return models

    def compute_all_cached(self, summaries: str, articles: str, source_filename: str) -> float:
        # Reading or writing mode, depending on cached file existence
        if os.path.exists(os.path.join(DIR, f"models/{self.task}_{self.language}/cached_{source_filename}.jsonl")):
            self.cache_mode = "r"
        else:
            self.cache_mode = "w"

        scores = []
        for i, (summary, article) in enumerate(zip(summaries, articles)):
            self.cache_index = i
            pre_answ_scores, pre_f1_scores, pre_ques_consists = self.compute_precision_scores(summary, article)
            rec_answ_scores, rec_f1_scores, rec_ques_consists, rec_weight_ws = self.compute_recall_scores(summary, article, source_filename)

            scores.append(
                self.compute_safeval_score(
                    (pre_answ_scores, pre_f1_scores, pre_ques_consists),
                    (rec_answ_scores, rec_f1_scores, rec_ques_consists, rec_weight_ws)
                )
            )

            # Writing line by line
            if self.cache_mode == "w":
                with open(os.path.join(DIR, f"models/{self.task}_{self.language}/cached_{source_filename}.jsonl"), "a") as cache:
                    cache.write(f"{json.dumps(self.cached_questions_article)}\n")

        self.cache_index = None
        self.cached_questions_article = None
        self.cache_mode = None

        return np.mean(scores).item()

    def compute_all(self, hypothesis: str, source: str = None, reference: str = None) -> dict:

        assert source != None or reference != None, "you need to provide at least the source or the reference"
        assert source != '', "the source should not be an empty string"
        assert reference != '', "the reference should not be an empty string"
        if hypothesis == '':
            return {'scores': {'fscore': 0 , 'precision': 0, 'recall': 0}}

        # Preprocessing
        if self.src_preproc_pipe is not None:
            if source is not None:
                source = self.src_preproc_pipe(source)

        keys = []
        if source:
            keys += ['src_hyp', 'hyp_src']
        if reference:
            keys += ['ref_hyp', 'hyp_ref']
            if self.do_regularize:
                keys += ['src_ref', 'ref_src']

        d_results = {'logs':{}, 'scores':{}}
        for key in keys:
            if key == 'src_hyp':
                inp_1, inp_2 = source, hypothesis
            elif key == 'hyp_src':
                inp_1, inp_2 = hypothesis, source
            elif key == 'ref_hyp':
                inp_1, inp_2 = reference, hypothesis
            elif key == 'hyp_ref':
                inp_1, inp_2 = hypothesis, reference
            elif key == 'src_ref':
                inp_1, inp_2 = source, reference
            elif key == 'ref_src':
                inp_1, inp_2 = reference, source

            d_results['logs'][key] = self.compute_scores(inp_1, inp_2, key=key)

            weighter_probs = None
            if key == 'src_hyp' and self.models['Weighter'] is not None:
                weighter_probs = self.compute_weighter(d_results['logs'][key]['questions'],
                                                       d_results['logs'][key]['answers'],
                                                       d_results['logs'][key]['sent_idxs'],
                                                       source)

            d_results['logs'][key]['weighter_probs'] = weighter_probs


        precision, recall, fscore = self.flat_score(d_results['logs'])
        d_results['scores']['fscore'] = fscore
        d_results['scores']['precision'] = precision
        d_results['scores']['recall'] = recall

        return d_results

    def compute_weighter(self, questions, asws, sents_idxs, source_doc):

        #Neutral Policy
        if self.models['Weighter'] is None:
            result_clf_probs = [1 for _ in questions]
        else:
            clf_inputs = [f'{idx} {self.sep} {asw} {self.sep} {question} {self.sep} {source_doc}' for idx, asw, question in zip(sents_idxs, asws, questions)]
            clf_probs, clf_texts = self.models['Weighter'].predict(clf_inputs)

            if len(clf_inputs) == 1:
                clf_probs['max'] = [clf_probs['max']]

            # when used as a classifier, only the first token prob is enough to evaluate the probability.
            result_clf_probs = [clf_prob[0] if clf_text == "true" else 1 - clf_prob[0]
                                for clf_text, clf_prob in zip(clf_texts, clf_probs['max'])]

        assert len(result_clf_probs) == len(questions)
        return result_clf_probs

    def compute_scores(self, text_qg, text_qa, key):

        key_1, key_2 = key.split('_')

        model_QG = self.models[key_1]['QG']
        questions, asws, sents_idxs = self.generate_relevant_question(text_qg, model_QG, key_1)

        model_QA = self.models[key_2]['QA']
        answerability_scores, f1_scores, bert_scores = self.compute_qa_scores(text_qa, questions, asws, model_QA)

        question_consistencies = None
        if self.do_consistency:
            model_QA_consistency = self.models[key_1]['QA']
            question_consistencies = self.filter_questions(text_qg, questions, asws, model_QA_consistency)

        logs = dict()
        logs.update({
            'text_qg': text_qg,
            'text_qa': text_qa,
            'questions': questions,
            'answers': asws,
            'sent_idxs': sents_idxs,
            'question_consistencies': question_consistencies,
            'answerability_scores': answerability_scores,
            'f1_scores': f1_scores,
            'bert_scores': bert_scores
        })

        return logs

    def base_score(self, answ_scores, f1_scores, bert_scores, ques_consists, weighter_probs=None):

        def regularizer(f1_scores, answ_scores, bert_scores, list_reg):

            assert len(f1_scores) == len(list_reg)

            f1_scores = np.multiply(f1_scores, list_reg).tolist()
            answ_scores = np.multiply(answ_scores, list_reg).tolist()
            if bert_scores is not None:
                bert_scores = np.multiply(bert_scores, list_reg).tolist()

            return f1_scores, answ_scores, bert_scores

        score, penalty = 0, 0

        if len(f1_scores) > 0:

            # sometimes the answers scores return a value ~1.000000X which is superior to 1
            list_borned = lambda a_list: [max(min(1, x), 0) for x in a_list]
            f1_scores = list_borned(f1_scores)
            answ_scores = list_borned(answ_scores)
            if self.do_BERTScore:
                assert bert_scores is not None, 'bert_scores is None while do_BERTScore is activated'
                bert_scores = list_borned(bert_scores)

            if self.do_consistency:
                assert ques_consists is not None, "ques_consists is None. Please compute the score with ques_consists activate."
                f1_scores, answ_scores, bert_scores = regularizer(f1_scores, answ_scores, bert_scores, ques_consists)

            if self.do_weighter and weighter_probs != None:
                f1_scores, answ_scores, bert_scores = regularizer(f1_scores, answ_scores, bert_scores, weighter_probs)

            scores = f1_scores + answ_scores
            if self.do_BERTScore:
                assert bert_scores is not None, 'BERTScore is None while do_BERTScore is activated'
                scores += bert_scores

            score = np.average(scores)
            penalty = self.lambda_penalty * (1 - min(answ_scores))

        assert 0 <= score <= 1, "score should be in [0-1] "
        return score, penalty

    def flat_score(self, d_logs):

        rec_scores, rec_penalties, prec_scores, prec_penalties,ref_scores, ref_penalties = [], [], [], [], [], []
        for key_name, key_dict in (d_logs.items()):

            score, pen = self.base_score(key_dict['answerability_scores'],
                                         key_dict['f1_scores'],
                                         key_dict['bert_scores'],
                                         key_dict['question_consistencies'],
                                         key_dict['weighter_probs'])

            if key_name in ['src_hyp', 'ref_hyp']:
                rec_scores.append(score)
                rec_penalties.append(pen)
            elif key_name in ['hyp_src', 'hyp_ref']:
                prec_scores.append(score)
                prec_penalties.append(pen)
            elif key_name in ['src_ref', 'ref_src']:
                ref_scores.append(score)
                ref_penalties.append(pen)

        recall = np.average(rec_scores) + sum(rec_penalties)
        precision = np.average(prec_scores) + sum(prec_penalties)
        f_score = (recall + precision)/2

        if self.do_regularize and len(ref_scores)>0:
            ref_score = np.average(ref_scores) + sum(ref_penalties)
            f_score = max(f_score / (ref_score + 1e-6), 1)  # clipping

        return precision, recall, f_score

    def generate_relevant_question(self, text, model_QG, key):

        def predict_questions(batch_sents, batch_asws):
            question_probs, question_texts = model_QG.predict(batch_sents, beam_size=self.qg_beam_size)
            # extend wrt beam size:
            batch_asws = [batch_asws[idx] for idx in range(len(batch_asws)) for _ in range(self.qg_beam_size)]
            return question_texts, batch_asws

        questions, answers, list_sents_idx = [], [], []
        batch_sents, batch_asws = [], []
        for i_sent, sent in enumerate(self.sentencize(text)):

            if i_sent == self.limit_sent:
                break

            mode_asw = 'text'
            if key == 'src' and self.task == "data2text":
                mode_asw = 'data'
            sent_asws = self.get_all_answers(sent, mode_asw=mode_asw)

            for i_asw, asw in enumerate(sent_asws):
                str_prefix = ''
                if self.qg_prefix is not None:
                    str_prefix = f'{self.qg_prefix} {self.sep} '
                batch_sents.append(f'{str_prefix}{asw} {self.sep} {sent}')
                batch_asws.append(asw)
                list_sents_idx += [i_sent] * self.qg_beam_size
                if len(batch_sents) == self.qg_batch_size:
                    new_questions, new_answers = predict_questions(batch_sents, batch_asws)
                    answers += new_answers
                    questions += new_questions
                    batch_sents, batch_asws = [], []

        if len(batch_sents) > 0:
            new_questions, new_answers = predict_questions(batch_sents, batch_asws)
            answers += new_answers
            questions += new_questions
        if len(questions) == 0:
            pass

        assert (len(questions) == len(answers) and len(questions) == len(list_sents_idx))
        return questions, answers, list_sents_idx

    def get_all_answers(self, sent, mode_asw='text'):

        sent_asws = []

        if mode_asw=='text':
            s = self.spacy_pipeline(sent)
            if 'NER' in self.answer_types:
                sent_asws += [a.text for a in s.ents]
            if 'NOUN' in self.answer_types:
                sent_asws += [a.text for a in s.noun_chunks]
            if 'SPANER' in self.answer_types:
                pass  # todo not implemented
        elif mode_asw == 'data':

            asw_toks = []
            is_asw = False
            for tok in sent.split():

                if tok == ']':
                    sent_asws.append(' '.join(asw_toks))
                    is_asw = False
                    asw_toks = []

                if is_asw:
                    asw_toks.append(tok)

                if tok == '[':
                    is_asw = True
        else:
            raise ('Situation not handle.')
        return sent_asws

    def filter_questions(self, text, questions, answers, model_QA):
        qa_inputs = [f'{question} {self.sep} {text}' for question, asw in zip(questions, answers)]
        qa_scores, qa_texts = model_QA.predict(qa_inputs)
        f1_scores = [calculate_f1_squad(qa_text, asw) for qa_text, asw in zip(qa_texts, answers)]
        assert len(questions) == len(f1_scores)
        return f1_scores

    def compute_qa_scores(self, questioned_text, questions, asws, model_QA):

        qa_inputs = [f'{question} {self.sep} {questioned_text}' for question, asw in zip(questions, asws)]
        qa_scores, qa_texts = model_QA.predict(qa_inputs)

        answerability_scores = [1 - qa_score[0] for qa_score in qa_scores[self.keep_score_idx]]
        f1_scores = [calculate_f1_squad(qa_text, asw) for qa_text, asw in zip(qa_texts, asws)]
        bert_scores = self.calculate_BERTScore(qa_texts, asws)
        return answerability_scores, f1_scores, bert_scores

    def sentencize(self, text):
        preprocessed_context = self.spacy_pipeline(text)
        return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]

    def calculate_BERTScore(self, model_predictions, gold_references):

        if self.do_BERTScore == None:
            return None

        if len(model_predictions) == 0:
            return []
        self.metric_BERTScore.add_batch(predictions=model_predictions, references=gold_references)
        final_score = self.metric_BERTScore.compute(model_type='bert-base-multilingual-cased', device=self.device)

        # set all unanswerable scores to 0
        for i, (pred) in enumerate(model_predictions):
            if pred == "unanswerable":
                final_score['f1'][i] = 0.0

        return final_score['f1']




def load_t2t_model(path_model, device, keep_score_idx=[], model_batch_size=48):
    return APIQA(
        pretrained_model_name_or_path=path_model,
        max_source_length=512,
        model_batch_size=model_batch_size,
        keep_score_idx=keep_score_idx,
        device=device
    )


def load_t2t_uni_model(
    path_model,
    device,
    keep_score_idx=[],
    model_batch_size=48,
):
    isCuda = device != 'cpu'
    return T2tUniModel(
        path_model,
        isCuda,
        keep_score_idx,
        model_batch_size,
    )
