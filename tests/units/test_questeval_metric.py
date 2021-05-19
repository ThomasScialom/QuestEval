import pytest

from questeval.questeval_metric import QuestEval
from tests.utils.helpers import compute_questeval_score
from tests.resources.constants import (
    SRC_1, REF_1, HYP_1, RES_1,
    SRC_2, REF_2, HYP_2, RES_2,
    SRC_3, REF_3, HYP_3, RES_3,
    SRC_multi_1, REF_multi_1, HYP_multi_1, RES_multi_1,
    SRC_sum_1, REF_sum_1, HYP_sum_1, RES_sum_1,
    SRC_sum_2, REF_sum_2, HYP_sum_2, RES_sum_2,
    SRC_D2T_1, REF_D2T_1, HYP_D2T_1, RES_D2T_1

)

def test_questeval_exceptions():

    questeval = QuestEval()

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1, HYP=None, SRC=None, REF=None)

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1,HYP="", SRC=None, REF=None)

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1,HYP="", SRC="", REF="")

    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval, res=-1,HYP="This is a test", SRC="Test", REF="")


def test_questeval_metric_text2text():

    questeval = QuestEval()

    # default example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_1['source_reference'], HYP=HYP_1, SRC=SRC_1, REF=REF_1)

    # default example - source only
    compute_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=SRC_1, REF=None)

    # default example - reference only
    compute_questeval_score(questeval=questeval, res=RES_1['reference'], HYP=HYP_1, SRC=None, REF=REF_1)

    # default example - check that source and reference can be switched
    compute_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=None, REF=SRC_1)

    # default example - check that source and reference are equally weighted
    compute_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=SRC_1, REF=SRC_1)

    # README example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_2['source_reference'], HYP=HYP_2, SRC=SRC_2, REF=REF_2)

    # README example - source only
    compute_questeval_score(questeval=questeval, res=RES_2['source'], HYP=HYP_2, SRC=SRC_2, REF=None)

    # README example - reference only
    compute_questeval_score(questeval=questeval, res=RES_2['reference'], HYP=HYP_2, SRC=None, REF=REF_2)

    # Long example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_3['source_reference'], HYP=HYP_3, SRC=SRC_3, REF=REF_3)

    # Long example - source only
    compute_questeval_score(questeval=questeval, res=RES_3['source'], HYP=HYP_3, SRC=SRC_3, REF=None)

    # Long example - reference only
    compute_questeval_score(questeval=questeval, res=RES_3['reference'], HYP=HYP_3, SRC=None, REF=REF_3)

def test_questeval_metric_text2text_mutlilingual():

    questeval = QuestEval(language='multi')

    # French example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_multi_1['source_reference'], HYP=HYP_multi_1, SRC=SRC_multi_1, REF=REF_multi_1)

    # French example - source only
    compute_questeval_score(questeval=questeval, res=RES_multi_1['source'], HYP=HYP_multi_1, SRC=SRC_multi_1, REF=None)

    # French example - reference only
    compute_questeval_score(questeval=questeval, res=RES_multi_1['reference'], HYP=HYP_multi_1, SRC=None, REF=REF_multi_1)

def test_questeval_metric_summarization():

    questeval = QuestEval(task="summarization", do_weighter=True)

    # Summarization example 1 - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum_1['source_reference_with_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=REF_sum_1)

    # Summarization example 1 - source
    compute_questeval_score(questeval=questeval, res=RES_sum_1['source_with_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=None)

    # Summarization example 1 - reference
    compute_questeval_score(questeval=questeval, res=RES_sum_1['reference'], HYP=HYP_sum_1, SRC=None, REF=REF_sum_1)

    # Summarization example 2 - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum_2['source_reference_with_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=REF_sum_2)

    # Summarization example 2 - source
    compute_questeval_score(questeval=questeval, res=RES_sum_2['source_with_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=None)

    # Summarization example 2 - reference
    compute_questeval_score(questeval=questeval, res=RES_sum_2['reference'], HYP=HYP_sum_2, SRC=None, REF=REF_sum_2)

    # If we remove the weighter
    questeval.do_weighter = False

    # Summarization example 1 - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum_1['source_reference_no_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=REF_sum_1)

    # Summarization example 1 - source
    compute_questeval_score(questeval=questeval, res=RES_sum_1['source_no_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=None)

    # Summarization example 1 - reference
    compute_questeval_score(questeval=questeval, res=RES_sum_1['reference'], HYP=HYP_sum_1, SRC=None, REF=REF_sum_1)

    # Summarization example 2 - source and reference
    compute_questeval_score(questeval=questeval, res=RES_sum_2['source_reference_no_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=REF_sum_2)

    # Summarization example 2 - source
    compute_questeval_score(questeval=questeval, res=RES_sum_2['source_no_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=None)

    # Summarization example 2 - reference
    compute_questeval_score(questeval=questeval, res=RES_sum_2['reference'], HYP=HYP_sum_2, SRC=None, REF=REF_sum_2)


def test_questeval_metric_data2text():

    questeval = QuestEval(task="webnlg")

    # Checking the source linearization
    with pytest.raises(AssertionError) as loc_error:
        compute_questeval_score(questeval=questeval, res=RES_D2T_1['source_reference'], HYP=HYP_D2T_1[0], SRC=SRC_D2T_1, REF=REF_D2T_1)

    # Data2text example - source and reference
    compute_questeval_score(questeval=questeval, res=RES_D2T_1['source_reference'], HYP=HYP_D2T_1, SRC=SRC_D2T_1, REF=REF_D2T_1)

    # Data2text example - source
    compute_questeval_score(questeval=questeval, res=RES_D2T_1['source'], HYP=HYP_D2T_1, SRC=SRC_D2T_1, REF=None)

    # Data2text example - reference
    compute_questeval_score(questeval=questeval, res=RES_D2T_1['reference'], HYP=HYP_D2T_1, SRC=None, REF=REF_D2T_1)
