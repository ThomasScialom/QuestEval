import os
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'questeval'
DESCRIPTION = "Supporting code for QuestEval metric."
URL = 'https://github.com/recitalAI/QuestEval'
EMAIL = 't.scialom@gmail.com'
AUTHOR = 'Thomas Scialom'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.1'

REQUIRED = [
    'spacy==3.0.0',
    'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz',
    'xx_ent_wiki_sm @ https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.0.0/xx_ent_wiki_sm-3.0.0.tar.gz',
    'transformers==3.0.1',
    'datasets>=1.2.1',
    'bert_score==0.3.8',
    'pyarrow==0.17.1',
    'Unidecode==1.2.0',
]

EXTRAS = {
}

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Installing a part of the repo
os.system("pip install -e unilm/s2s-ft")

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["examples"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
