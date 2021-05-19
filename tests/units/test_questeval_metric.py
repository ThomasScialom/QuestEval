import pytest

from questeval.questeval_metric import QuestEval
from tests.utils.helpers import test_questeval_score
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
        questeval.compute_all(hypothesis=None, source=None, reference=None)

    with pytest.raises(AssertionError) as loc_error:
        questeval.compute_all(hypothesis="", source=None, reference=None)

    with pytest.raises(AssertionError) as loc_error:
        questeval.compute_all(hypothesis="", source="", reference="")

    with pytest.raises(AssertionError) as loc_error:
        questeval.compute_all(hypothesis="This is a test", source="Test", reference="")


def test_questeval_metric_text2text():

    questeval = QuestEval()

    # default example - source and reference
    test_questeval_score(questeval=questeval, res=RES_1['source_reference'], HYP=HYP_1, SRC=SRC_1, REF=REF_1)

    # default example - source only
    test_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=SRC_1, REF=None)

    # default example - reference only
    test_questeval_score(questeval=questeval, res=RES_1['reference'], HYP=HYP_1, SRC=None, REF=REF_1)

    # default example - check that source and reference can be switched
    test_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=None, REF=SRC_1)

    # default example - check that source and reference are equally weighted
    test_questeval_score(questeval=questeval, res=RES_1['source'], HYP=HYP_1, SRC=SRC_1, REF=SRC_1)

    # README example - source and reference
    test_questeval_score(questeval=questeval, res=RES_2['source_reference'], HYP=HYP_2, SRC=SRC_2, REF=REF_2)

    # README example - source only
    test_questeval_score(questeval=questeval, res=RES_2['source'], HYP=HYP_2, SRC=SRC_2, REF=None)

    # README example - reference only
    test_questeval_score(questeval=questeval, res=RES_2['reference'], HYP=HYP_2, SRC=None, REF=REF_2)

    # Long example - source and reference
    test_questeval_score(questeval=questeval, res=RES_3['source_reference'], HYP=HYP_3, SRC=SRC_3, REF=REF_3)

    # Long example - source only
    test_questeval_score(questeval=questeval, res=RES_3['source'], HYP=HYP_3, SRC=SRC_3, REF=None)

    # Long example - reference only
    test_questeval_score(questeval=questeval, res=RES_3['reference'], HYP=HYP_3, SRC=None, REF=REF_3)

def test_questeval_metric_text2text_mutlilingual():

    questeval = QuestEval(isCuda=True, language='multi')

    # French example - source and reference
    test_questeval_score(questeval=questeval, res=RES_multi_1['source_reference_with_weighter'], HYP=HYP_multi_1, SRC=SRC_multi_1, REF=REF_multi_1)

    # French example - source only
    test_questeval_score(questeval=questeval, res=RES_multi_1['source_with_weighter'], HYP=HYP_multi_1, SRC=SRC_multi_1, REF=None)

    # French example - reference only
    test_questeval_score(questeval=questeval, res=RES_multi_1['reference'], HYP=HYP_multi_1, SRC=None, REF=REF_multi_1)

def test_questeval_metric_summarization():

    questeval = QuestEval(isCuda=True, language='en', task="summarization")

    # Summarization example 1 - source and reference
    test_questeval_score(questeval=questeval, res=RES_sum_1['source_reference_with_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=REF_sum_1)

    # Summarization example 1 - source
    test_questeval_score(questeval=questeval, res=RES_sum_1['source_with_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=None)

    # Summarization example 1 - reference
    test_questeval_score(questeval=questeval, res=RES_sum_1['reference'], HYP=HYP_sum_1, SRC=None, REF=REF_sum_1)

    # Summarization example 2 - source and reference
    test_questeval_score(questeval=questeval, res=RES_sum_2['source_reference_with_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=REF_sum_2)

    # Summarization example 2 - source
    test_questeval_score(questeval=questeval, res=RES_sum_2['source_with_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=None)

    # Summarization example 2 - reference
    test_questeval_score(questeval=questeval, res=RES_sum_2['reference'], HYP=HYP_sum_2, SRC=None, REF=REF_sum_2)

    # If we remove the weighter
    questeval.do_weighter = False

    # Summarization example 1 - source and reference
    test_questeval_score(questeval=questeval, res=RES_sum_1['source_reference_no_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=REF_sum_1)

    # Summarization example 1 - source
    test_questeval_score(questeval=questeval, res=RES_sum_1['source_no_weighter'], HYP=HYP_sum_1, SRC=SRC_sum_1, REF=None)

    # Summarization example 1 - reference
    test_questeval_score(questeval=questeval, res=RES_sum_1['reference'], HYP=HYP_sum_1, SRC=None, REF=REF_sum_1)

    # Summarization example 2 - source and reference
    test_questeval_score(questeval=questeval, res=RES_sum_2['source_reference_no_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=REF_sum_2)

    # Summarization example 2 - source
    test_questeval_score(questeval=questeval, res=RES_sum_2['source_no_weighter'], HYP=HYP_sum_2, SRC=SRC_sum_2, REF=None)

    # Summarization example 2 - reference
    test_questeval_score(questeval=questeval, res=RES_sum_2['reference'], HYP=HYP_sum_2, SRC=None, REF=REF_sum_2)


def test_questeval_metric_data2text():

    questeval = QuestEval(isCuda=True, task="webnlg")

    # Checking the source linearization
    with pytest.raises(AssertionError) as loc_error:
        test_questeval_score(questeval=questeval, res=RES_D2T_1['source_reference'], HYP=HYP_D2T_1[0], SRC=SRC_D2T_1, REF=REF_D2T_1)

    # Data2text example - source and reference
    test_questeval_score(questeval=questeval, res=RES_D2T_1['source_reference'], HYP=HYP_D2T_1, SRC=SRC_D2T_1, REF=REF_D2T_1)

    # Data2text example - source
    test_questeval_score(questeval=questeval, res=RES_D2T_1['source'], HYP=HYP_D2T_1, SRC=SRC_D2T_1, REF=None)

    # Data2text example - reference
    test_questeval_score(questeval=questeval, res=RES_D2T_1['reference'], HYP=HYP_D2T_1, SRC=None, REF=REF_D2T_1)
