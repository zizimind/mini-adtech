"""Deterministic benchmark runner producing artifact metrics."""

import json
import os
import random
from collections import defaultdict
from statistics import mean
from time import perf_counter

from mini_adtech import (
    ADS,
    AD_QUERIES,
    AD_USERS,
    ItemIndex,
    ShadowScorer,
    Throttler,
    TokenBucket,
    FrequencyCap,
    PATIENT_QUERIES,
    PATIENTS,
    HEALTH_PLANS,
    LinearRanker,
    recommend_topk,
    run_auction,
)


def _run_adtech_benchmark(n_requests: int, seed: int) -> dict:
    if n_requests <= 0:
        raise ValueError("n_requests must be > 0 for adtech benchmark")
    random.seed(seed)
    index = ItemIndex(ADS)
    shadow = ShadowScorer()
    sim_day = 3_600.0
    dt = sim_day / n_requests
    buckets = {ad["advertiser"]: TokenBucket(ad["budget"], sim_day) for ad in ADS}
    fcap = FrequencyCap(max_per_session=2)
    throttler = Throttler(target_win_rate=0.20)

    spent = defaultdict(float)
    wins = defaultdict(int)
    nofill = 0
    latencies_ms = []

    for _ in range(n_requests):
        t0 = perf_counter()
        user = random.choice(AD_USERS)
        query = random.choice(AD_QUERIES)

        for bucket in buckets.values():
            bucket.tick(dt)

        raw = index.search(query, top_k=5)
        if not raw:
            nofill += 1
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue

        # Request-level throttling: decide once whether to participate
        # in this auction request (mirrors simulate_adtech behavior).
        if not throttler.should_bid():
            nofill += 1
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue

        candidates = []
        for item, tfidf in raw:
            adv = item["advertiser"]
            if not buckets[adv].consume():
                continue
            if not fcap.allowed(user["id"], adv):
                continue
            champ_ecpm, _ = shadow.score(item, tfidf, user, use_ecpm=True)
            candidates.append((item, tfidf, champ_ecpm))

        if not candidates:
            throttler.record(won=False)
            nofill += 1
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue

        winner, price = run_auction(candidates, reserve_cpm=500.0)
        if winner is None:
            throttler.record(won=False)
            nofill += 1
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue

        clicked = random.random() < winner["quality"] * 0.30
        label = 1.0 if clicked else 0.0
        wt = next(score for it, score, _ in candidates if it["id"] == winner["id"])
        shadow.update(winner, wt, user, label)
        fcap.record(user["id"], winner["advertiser"])
        throttler.record(won=True)
        spent[winner["advertiser"]] += price
        wins[winner["advertiser"]] += 1
        latencies_ms.append((perf_counter() - t0) * 1000)

    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[int(0.50 * len(sorted_lat))]
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]
    total_revenue = sum(spent.values())
    delivered = n_requests - nofill
    rpm = (total_revenue * 1000 / delivered) if delivered > 0 else 0.0

    return {
        "requests": n_requests,
        "nofill_rate": nofill / n_requests,
        "total_revenue": total_revenue,
        "rpm": rpm,
        "avg_wins_per_advertiser": mean(wins.values()) if wins else 0.0,
        "latency_ms": {"p50": p50, "p95": p95},
    }


def _run_health_benchmark(n_requests: int, seed: int) -> dict:
    if n_requests <= 0:
        raise ValueError("n_requests must be > 0 for health benchmark")
    random.seed(seed)
    index = ItemIndex(HEALTH_PLANS)
    ranker = LinearRanker(lr=0.05)
    latencies_ms = []
    top1_scores = []
    enrollments = 0

    for _ in range(n_requests):
        t0 = perf_counter()
        patient = random.choice(PATIENTS)
        query = random.choice(PATIENT_QUERIES)
        raw = index.search(query, top_k=6)
        if not raw:
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue
        candidates = []
        for plan, tfidf in raw:
            score = ranker.fit_score(plan, tfidf, patient)
            candidates.append((plan, tfidf, score))
        top3 = recommend_topk(candidates, k=3)
        if not top3:
            latencies_ms.append((perf_counter() - t0) * 1000)
            continue
        top1_scores.append(top3[0][2])
        enrolled = random.random() < top3[0][0]["quality"] * 0.40
        enrollments += 1 if enrolled else 0
        ranker.update(top3[0][0], top3[0][1], patient, 1.0 if enrolled else 0.0)
        latencies_ms.append((perf_counter() - t0) * 1000)

    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[int(0.50 * len(sorted_lat))]
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]
    return {
        "requests": n_requests,
        "enrollment_rate": enrollments / n_requests,
        "avg_top1_fit_score": mean(top1_scores) if top1_scores else 0.0,
        "latency_ms": {"p50": p50, "p95": p95},
    }


def run_benchmark(ad_requests: int = 500, health_requests: int = 300) -> dict:
    """Run deterministic benchmark suite and return metrics."""
    adtech = _run_adtech_benchmark(ad_requests, seed=11)
    health = _run_health_benchmark(health_requests, seed=19)
    return {"adtech": adtech, "health": health}


def main() -> None:
    metrics = run_benchmark()
    os.makedirs("artifacts", exist_ok=True)
    output_path = "artifacts/benchmark.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"Wrote benchmark artifact to {output_path}")


if __name__ == "__main__":
    main()
