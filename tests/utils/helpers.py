
def test_questeval_score(questeval, res, HYP, SRC, REF):
    score = questeval.compute_all(hypothesis=HYP, source=SRC, reference=REF)
    for k in score['scores'].keys():
        assert round(score['scores'][k], 4) == round(res[k], 4)