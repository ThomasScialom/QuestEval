# QuestEval

QuestEval is a **NLG metric** to assess if two different inputs contain the same information. The metric, based on Question Generation and Answering can deal with **multimodal** and **multilingual** inputs. 
It is the result from an (on-going) international collaboration, and so far it tackles various tasks:

- [Summarization](#summarization)
- [Text Simplification](#text-simplification)
- [Data2text](#data2text)
- [Image Captioning](#image-captioning)
- [Multilingual Evaluation](#multilingual-evaluation)


Planned extensions: 
- Machine Translation
- Multi-references option
- Computational optimisation (batch on multiple examples, generated questions caching)
- Hyperparameters hashcode for results reproducibility
- Migration to the last Hugging Face Transformers 
- Ability to use your own models

## 1/ Installing QuestEval
```
$ conda create --name questeval python=3.7
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

The pre-trained QA/QG models will be automatically downloaded. Alternatively, you can use your own models.

## 2/ Using QuestEval 

The default task is *text2text* and the default language is *en*. It allows to measure the content similarity between any English texts:

```
from questeval.questeval_metric import QuestEval

source = """After wildfires consumed an entire town, students and teachers who had planned for remote classes found some comfort in staying connected amid the chaos."""
hypothesis = """Ash fell from an apocalyptic orange sky as Jennifer Willin drove home last week from the only school in tiny Berry Creek, Calif., where she had picked up a pair of Wi-Fi hot spots for her daughters’ remote classes. Hours later, her cellphone erupted with an emergency alert: Evacuate immediately. By the next morning, what one official described as a “massive wall of fire” had swept through the entire Northern California town of about 1,200 people, killing nine residents, including a 16-year-old boy, and destroying the school and almost every home and business. Ms. Willin and her family escaped to a cramped hotel room 60 miles away. In her panic, she had forgotten to grab masks, but she had the hot spots, along with her daughters’ laptops and school books. On Monday, the two girls plan to meet with their teachers on Zoom, seeking some comfort amid the chaos. They’re still able to be in school,” Ms. Willin said, “even though the school burned to the ground.” As the worst wildfire season in decades scorches the West amid a still raging pandemic, families and educators who were already starting the strangest and most challenging school year of their lifetimes have been traumatized all over again. Tens of thousands of people have been forced to flee their homes, with some mourning the loss of their entire communities. But amid the twin disasters, the remote learning preparations that schools made for the coronavirus are providing a strange modicum of stability for teachers and students, letting many stay connected and take comfort in an unexpected form of virtual community."""

questeval = QuestEval()
score = questeval.compute_all(hypothesis, source)
print(score['scores'])
```
Output:
```
{fscore': 0.2883088133952934, 
'precision': 0.5038477301470266, 
'recall': 0.07276989664356022}
```

Yes, it works without any references, but if the reference is available, you can also add it:
```
reference = """After wildfires consumed the town, students who had planned for remote classes found some comfort in staying connected amid the chaos."""
score = questeval.compute_all(hypothesis, source, reference)
print(score['scores'])
```
Output:
```
{'fscore': 0.4750318370987159, 
'precision': 0.5820995386296233, 
'recall': 0.36796413556780855}
```
Note that the score is always between 0 and 1.

Alternatively, you can compute the score by comparing the evaluated text only to the reference: 
```
score = questeval.compute_all(hypothesis, None, reference)
print(score['scores'])
```
Output:
```
{'fscore': 0.6617548608021384, 
'precision': 0.66035134711222, 
'recall': 0.6631583744920568}
```
This means that **QuestEval can be used to evaluate any NLG task where references are available**.
For tasks specificities, see below. 

In addition, you can access all the logs including the generated questions and predicted answers. For instance, the generated questions on the source that were asked on the hypothesis are available via `score['logs']['src_hyp']['questions']`.

***[coming soon]*** We also provide more examples in the Jupyter notebook *example/Examples.ipynb*. The notebook also contains all the code to reproduce the results in the paper *QuestEval: Summarization Asks for Fact-based Evaluation*.

To run the notebook in your environment:

```
(questeval) $ conda install jupyter
(questeval) $ python -m ipykernel install --user --name=questeval
(questeval) $ pip install matplotlib
(questeval) $ pip install ipywidgets
```

## 3/ Tasks specificities

### Summarization
The project is a collaboration work between [LIP6 Lab](https://mlia.lip6.fr/), [New York University](https://wp.nyu.edu/ml2/) and [ReciTAL Research](https://recital.ai/en/research-development/).

QuestEval also handles summarization specificities: we developped a Weighter that selects only the questions asking about the important facts that are worth to be summarized. Read more in the original [paper](https://arxiv.org/abs/2103.12693). To activate this Weighter `do_weighter=True` when loading the metric.

Paper: [QuestEval: Summarization Asks for Fact-based Evaluation](https://arxiv.org/abs/2103.12693)

How to cite:
```
@article{scialom2020QuestEval,
  title={QuestEval: Summarization Asks for Fact-based Evaluation},
  author={Scialom, Thomas and Dray, Paul-Alexis and Gallinari Patrick and Lamprier Sylvain and Piwowarski Benjamin and Staiano Jacopo and Wang Alex},
  journal={arXiv preprint arXiv:2103.12693},
  year={2021}
}
```

### Text Simplification

For Text Simplification, we recommend to activate the BERTScore for computing the similarity between two answers `do_BERTScore=True` when loading the metric. It ranks better the systems than BLEU or SARI metrics as reported in the paper.

Paper: [Rethinking Automatic Evaluation in Sentence Simplification](https://arxiv.org/abs/2104.07560)

How to cite:
```
@article{scialom2021rethinking,
  title={Rethinking Automatic Evaluation in Sentence Simplification},
  author={Scialom, Thomas and Martin, Louis and Staiano, Jacopo and de la Clergerie, {\'E}ric Villemonte and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2104.07560},
  year={2021}
}
```
### Data2text

We propose by default trained QA/QG models dealing with table inputs (e.g. E2E or Webnlg, see more in coming very soon). To load QuestEval for data2text tasks, specify *task=e2e* or *task=webnlg*. Note that you need a specific processing to linearised the tables. By default we handle the [GEM](https://gem-benchmark.com/) format for these two datasets. If you need an other preprocessing of the table, you can pass your custom function to Questeval: *src_preproc_pipe=custom_formating*.

Paper link: [Data-QuestEval: A Referenceless Metric for Data to Text Semantic Evaluation](https://arxiv.org/abs/2104.07555)

How to cite:
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

QuestEval supports non english evaluation: the parameter `language` can be set to `multi` to tackle non-English texts. For Question Generation and Answering we used the [m-Minilm model](https://github.com/microsoft/unilm/tree/master/minilm). For the answer selection, spacy does not support multilingual *noun chuncking*. For this reason, QuestEval can be less effective than its English version: **we are working on that!**


