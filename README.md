# QuestEval
![GitHub](https://img.shields.io/github/license/ThomasScialom/QuestEval)
![release](https://img.shields.io/github/v/release/ThomasScialom/QuestEval)
 
QuestEval is a **NLG metric** to assess if two different inputs contain the same information. The metric, based on Question Generation and Answering can deal with **multimodal** and **multilingual** inputs. 
It is the result from an (on-going) international collaboration, and so far it tackles various tasks:

- [Summarization](#summarization)
- [Text Simplification](#text-simplification)
- [Data2text](#data2text)
- [Image Captioning](#image-captioning)
- [Multilingual Evaluation](#multilingual-evaluation)


Planned extensions: 
- Multilingual Evaluation

## 1/ Installing QuestEval
```
$ conda create --name questeval python=3.9
$ conda activate questeval
```
**WARNING**: You need to install, before any package, correct version of [pytorch](https://pytorch.org/get-started/locally/#start-locally) linked to your cuda version.
```
(questeval) $ conda install pytorch cudatoolkit=10.1 -c pytorch
```

```
(questeval) $ conda install pip
(questeval) $ pip install -e .
```

## 2/ Using QuestEval 

The default `task` is `text2text` and the default `language` is `en`. It allows to measure the content similarity between any two English texts. This means that **QuestEval can be used to evaluate any NLG task where references are available**. Alternatively, we can compare the hyothesis to the source as detailed below.  
For tasks specificities, see below. 

Here is an example. Note that the code can take time since it requires generating and answering a set of questions. However, if you let the parameter `use_cache` to its default value at `True`, running the same example again will be very fast this time.

```
from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=True)

source_1 = "Since 2000, the recipient of the Kate Greenaway medal has also been presented with the Colin Mears award to the value of 35000."
prediction_1 = "Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award of the Kate Greenaway medal."
references_1 = [
    "Since 2000, the recipient of the Kate Greenaway Medal will also receive the Colin Mears Awad which worth 5000 pounds",
    "Since 2000, the recipient of the Kate Greenaway Medal has also been given the Colin Mears Award."
]

source_2 = "He is also a member of another Jungiery boyband 183 Club."
prediction_2 = "He also has another Jungiery Boyband 183 club."
references_2 = [
    "He's also a member of another Jungiery boyband, 183 Club.", 
    "He belonged to the Jungiery boyband 183 Club."
]

score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2],
    list_references=[references_1, references_2]
)

print(score)
```
Output:
```
{'corpus_score': 0.6115364039438322, 
'ex_level_scores': [0.5698804143875364, 0.6531923935001279]}
```

In the output, you have access to the `corpus_score` which corresponds to the average of each example score stored in `ex_level_scores`. Note that the score is always between 0 and 1.


### Reference-less mode

Yes, QuestEval can score a text without any references:

```
score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2]
)

print(score)
```
Output:
```
{'corpus_score': 0.5538727587335324, 
'ex_level_scores': [0.5382940950847808, 0.569451422382284]}
```

### Logs

You can have access to the logs containing all the information about the generated questions or the question answering outputs:
```
log = questeval.open_log_from_text(source_1)
```
For instance, to print the questions asked on `source_1`: 
```
print(log['asked'].keys())
```
Output:
```
dict_keys(['Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Me', 'What medal has been given to the winner of the Colin Mears award?', 'What has been given to the Colin Mears award since 2000?', 'What has been given to the winner of the Colin Mears award since 2000?', 'What has been given to the winner of the Kate Greenaway medal since 2000?'])
```

### Hash 

For reproducibility purpose, we defined a hash that contains exhaustive information such as the QuestEval version, as well as the used models names and the scores types:

```
questeval.__hash__()
```
Output:
```
"QuestEval_version=0.2.4_task=text2text_lang=en_preproc=None_consist=False_scores=('answerability', 'bertscore', 'f1')W_hash=None_hyp_QA_hash=ThomasNLG/t5-qa_squad2neg-en_ref_QA_hash=ThomasNLG/t5-qa_squad2neg-en_src_QA_hash=ThomasNLG/t5-qa_squad2neg-en_hyp_QG_hash=ThomasNLG/t5-qg_squad1-en_ref_QG_hash=ThomasNLG/t5-qg_squad1-en_src_QG_hash=ThomasNLG/t5-qg_squad1-en"
```

### Using your own models

The pre-trained QA/QG models will be automatically downloaded from Hugging Face. Alternatively, you can use your own models and change them dynamically with the `questeval.set_model(model_name)` method that takes as input a `path` or a Hugging Face `model_name`. If the `model_name` is hosted on the Hugging Face hub, it will be downloaded automatically.
You can also use any of your beloved models, the only required method to implement is the `model.predict()`.

## 3/ Tasks specificities

### Summarization

**Description:**  
The project is a collaboration work between [LIP6 Lab](https://mlia.lip6.fr/), [New York University](https://wp.nyu.edu/ml2/) and [ReciTAL Research](https://recital.ai/en/research-development/).  
QuestEval also handles summarization specificities: we developed a Weighter that selects only the questions asking about the important facts that are worth to be summarized. Read more in the original [paper](https://arxiv.org/abs/2103.12693). To activate this Weighter `do_weighter=True` when loading the metric. This parameter will be activated by default if `questeval.task=summarization`.  

**Warning:** the code has changed from the paper and Weigther input format is no longer correct. We plan to provide a new Weighter soon. However, in the meantime, if you plan to use it, we recommend to use the previous [version](https://github.com/recitalAI/QuestEval/releases/tag/v0.1.1).

**Paper:** [QuestEval: Summarization Asks for Fact-based Evaluation](https://arxiv.org/abs/2103.12693)

**How to cite:**
```
@article{scialom2020QuestEval,
  title={QuestEval: Summarization Asks for Fact-based Evaluation},
  author={Scialom, Thomas and Dray, Paul-Alexis and Gallinari Patrick and Lamprier Sylvain and Piwowarski Benjamin and Staiano Jacopo and Wang Alex},
  journal={arXiv preprint arXiv:2103.12693},
  year={2021}
}
```

### Text Simplification

**Description:**  
For Text Simplification, we recommend to use the default setup with the `text2text` task.  
QuestEval has been shown to perform better than BLEU, BERTScore or SARI metrics as reported in the paper.

**Paper:** [Rethinking Automatic Evaluation in Sentence Simplification](https://arxiv.org/abs/2104.07560)

**How to cite:**
```
@article{scialom2021rethinking,
  title={Rethinking Automatic Evaluation in Sentence Simplification},
  author={Scialom, Thomas and Martin, Louis and Staiano, Jacopo and de la Clergerie, {\'E}ric Villemonte and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2104.07560},
  year={2021}
}
```
### Data2text

**Description:**  
Trained data-QA/QG models that deals with input tables (e.g. E2E or Webnlg), are available by default. To load QuestEval for `data2text` tasks, specify `questeval.task=data2text`.  
Note that you need a specific preprocessing to linearize the tables. By default we handle the [GEM](https://gem-benchmark.com/) format. If you need an other preprocessing for your format, you can pass a custom function to Questeval: `src_preproc_pipe=custom_formating`. Importantly, the linearised input format needs to be consistent with respect to our own format.

**Paper:** [Data-QuestEval: A Referenceless Metric for Data to Text Semantic Evaluation](https://arxiv.org/abs/2104.07555)

**How to cite:**
```
@article{rebuffel2021data,
  title={Data-QuestEval: A Referenceless Metric for Data to Text Semantic Evaluation},
  author={Rebuffel, Cl{\'e}ment and Scialom, Thomas and Soulier, Laure and Piwowarski, Benjamin and Lamprier, Sylvain and Staiano, Jacopo and Scoutheeten, Geoffrey and Gallinari, Patrick},
  journal={arXiv preprint arXiv:2104.07555},
  year={2021}
}
```

### Image Captioning

*[Coming Soon]*

### Multilingual Evaluation

*[Coming Soon]*
