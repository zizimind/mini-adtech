"""Unit tests for mini_adtech edge cases and core behavior."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_adtech import (
    ADS,
    AD_USERS,
    ItemIndex,
    RAGEnricher,
    TokenBucket,
    recommend_topk,
    run_auction,
)


def test_item_index_search_unknown_term_returns_empty() -> None:
    index = ItemIndex(ADS)
    results = index.search(["nonexistent_term"], top_k=5)
    assert results == []


def test_item_index_top_k_non_positive_returns_empty() -> None:
    index = ItemIndex(ADS)
    assert index.search(["running"], top_k=0) == []
    assert index.search(["running"], top_k=-3) == []


def test_run_auction_rejects_negative_reserve() -> None:
    with pytest.raises(ValueError):
        run_auction([], reserve_cpm=-1.0)


def test_run_auction_returns_none_when_no_eligible_candidates() -> None:
    item = ADS[0]
    candidates = [(item, 0.1, 100.0)]
    winner, price = run_auction(candidates, reserve_cpm=500.0)
    assert winner is None
    assert price == 0.0


def test_run_auction_second_price_plus_increment() -> None:
    c1 = (ADS[0], 0.3, 1200.0)
    c2 = (ADS[1], 0.2, 900.0)
    winner, price = run_auction([c1, c2], reserve_cpm=500.0)
    assert winner["id"] == ADS[0]["id"]
    assert price == pytest.approx(0.901)


def test_recommend_topk_handles_non_positive_k() -> None:
    candidates = [(ADS[0], 0.2, 10.0), (ADS[1], 0.1, 5.0)]
    assert recommend_topk(candidates, k=0) == []
    assert recommend_topk(candidates, k=-1) == []


def test_token_bucket_negative_inputs_raise() -> None:
    with pytest.raises(ValueError):
        TokenBucket(daily_budget=-1.0)
    with pytest.raises(ValueError):
        TokenBucket(daily_budget=10.0, day_seconds=0.0)


def test_token_bucket_rejects_negative_tick_and_consume() -> None:
    bucket = TokenBucket(daily_budget=100.0, day_seconds=100.0)
    with pytest.raises(ValueError):
        bucket.tick(-1.0)
    with pytest.raises(ValueError):
        bucket.consume(-0.5)


def test_rag_parse_missing_reason_gets_default() -> None:
    raw = "INTENTS: find running shoes\nTAGS: running, shoes, sport\n"
    result = RAGEnricher._parse(raw)
    assert result["intents"] == ["find running shoes"]
    assert result["tags"] == ["running", "shoes", "sport"]
    assert result["reason"] == "No explicit reason provided by model."


def test_rag_rule_based_fallback_has_reason() -> None:
    item = ADS[0]
    user = AD_USERS[0]
    result = RAGEnricher._rule_based(item, user)
    assert result["intents"]
    assert result["tags"]
    assert isinstance(result["reason"], str)
    assert result["reason"] != ""
