
def compute_questeval_score(questeval, res, HYPS, SRCS, REFSS):
    d_score = questeval.corpus_questeval(hypothesis=HYPS, sources=SRCS, list_references=REFSS)
    assert round(d_score['corpus_score'], 4) - 1e-4 <= round(res['corpus_score'], 4) <= round(d_score['corpus_score'], 4) + 1e-4
    for seg_i, seg_score in enumerate(res['ex_level_scores']):
        assert 0 <= d_score['ex_level_scores'][seg_i] <= 1
        assert round(seg_score, 4) - 1e-4 <= round(d_score['ex_level_scores'][seg_i], 4) <= round(seg_score, 4) + 1e-4