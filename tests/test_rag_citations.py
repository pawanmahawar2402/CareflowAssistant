"""Citation parsing for grounded answers (M4.C3). No LLM, no Qdrant."""
from careflow.m4_rag.rag_agent import _CITE, _cited_ids, _norm_nums


def test_single_and_multi_id_brackets_are_both_parsed():
    text = "Gold members may self-refer [referral_policy#0, plan_gold_summary#0]. Fee is $50 [scheduling_and_cancellation#1]."
    assert _cited_ids(text) == ["referral_policy#0", "plan_gold_summary#0", "scheduling_and_cancellation#1"]


def test_chunk_indices_are_not_counted_as_numbers_in_the_answer():
    # the failing reply from the e2e run: '#0' must not be read as the number 0
    text = "Yes, Gold plan members may self-refer [referral_policy#0, plan_gold_summary#0]."
    assert _norm_nums(_CITE.sub("", text)) == set()


def test_quoted_amounts_are_still_checked():
    text = "The fee is $50 [scheduling_and_cancellation#1]."
    assert _norm_nums(_CITE.sub("", text)) == {"50"}
