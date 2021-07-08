import pytest

from questeval.questeval_metric import QuestEval
from tests.utils.helpers import compute_questeval_score
from tests.resources.constants import (
    SRC_empty, REF_empty, HYP_empty, RES_empty,
    SRC_t2t, REF_t2t, HYP_t2t, RES_t2t,
    SRC_D2T, REF_D2T, HYP_D2T, RES_D2T, SRC_D2T_wrong_format,
    SRC_sum, REF_sum, HYP_sum, RES_sum,
    SRC_multi_1, REF_multi_1, HYP_multi_1, RES_multi_1,
)

def test_questeval_exceptions():

    questeval = QuestEval()

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1,HYPS=[""], SRCS=None, REFSS=None)

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1,HYPS=None, SRCS=[""], REFSS=None)

def test_questeval_metric_text2text():

    questeval = QuestEval()

    # empty strings
    compute_questeval_score(questeval=questeval, res=RES_empty, HYPS=HYP_empty, SRCS=SRC_empty, REFSS=REF_empty)
    compute_questeval_score(questeval=questeval, res=RES_empty, HYPS=HYP_empty, SRCS=None, REFSS=REF_empty)
    compute_questeval_score(questeval=questeval, res=RES_empty, HYPS=HYP_empty, SRCS=SRC_empty, REFSS=None)

    # default example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_t2t['source_reference'], HYPS=HYP_t2t, SRCS=SRC_t2t, REFSS=REF_t2t)
    # default example - source only
    compute_questeval_score(questeval=questeval, res=RES_t2t['source'], HYPS=HYP_t2t, SRCS=SRC_t2t, REFSS=None)
    # default example - reference only
    compute_questeval_score(questeval=questeval, res=RES_t2t['reference'], HYPS=HYP_t2t, SRCS=None, REFSS=REF_t2t)
    # default example - check that source and reference can be switched
    compute_questeval_score(questeval=questeval, res=RES_t2t['source'], HYPS=HYP_t2t, SRCS=None, REFSS=[[s] for s in SRC_t2t])
    # default example - check that source and reference are equally weighted
    compute_questeval_score(questeval=questeval, res=RES_t2t['source'], HYPS=HYP_t2t, SRCS=SRC_t2t, REFSS=[[s] for s in SRC_t2t])

def test_questeval_metric_data2text():

    questeval = QuestEval(task="data2text")

    # Checking the source linearization
    from questeval.utils import WrongWebNlgFormat
    with pytest.raises(WrongWebNlgFormat) as loc_error:
        compute_questeval_score(questeval=questeval, res=RES_D2T['source_reference'], HYPS=HYP_D2T, SRCS=SRC_D2T_wrong_format, REFSS=REF_D2T)

    # Data2text example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_D2T['source_reference'], HYPS=HYP_D2T, SRCS=SRC_D2T, REFSS=REF_D2T)
    # Data2text example - source
    compute_questeval_score(questeval=questeval, res=RES_D2T['source'], HYPS=HYP_D2T, SRCS=SRC_D2T, REFSS=None)
    # Data2text example - reference
    compute_questeval_score(questeval=questeval, res=RES_D2T['reference'], HYPS=HYP_D2T, SRCS=None, REFSS=REF_D2T)

def test_questeval_metric_summarization():

    questeval = QuestEval(task="summarization", do_weighter=True)

    # Summarization - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum['source_reference'], HYPS=HYP_sum, SRCS=SRC_sum, REFSS=REF_sum)
    # Summarization - source
    compute_questeval_score(questeval=questeval, res=RES_sum['source'], HYPS=HYP_sum, SRCS=SRC_sum, REFSS=None)
    # Summarization - reference
    compute_questeval_score(questeval=questeval, res=RES_sum['reference'], HYPS=HYP_sum, SRCS=None, REFSS=REF_sum)

    # If we remove the weighter
    assert questeval.do_weighter == True
    questeval.do_weighter = False
    # Summarization - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum['source_reference_without_weighter'], HYPS=HYP_sum, SRCS=SRC_sum, REFSS=REF_sum)
    # Summarization - source
    compute_questeval_score(questeval=questeval, res=RES_sum['source_without_weighter'], HYPS=HYP_sum, SRCS=SRC_sum, REFSS=None)
    # Summarization - reference
    compute_questeval_score(questeval=questeval, res=RES_sum['reference_without_weighter'], HYPS=HYP_sum, SRCS=None, REFSS=REF_sum)
