"""
mini_adtech.py  —  A universal search, matching, ranking, and decision engine
with LangChain RAG enrichment. Pure Python core, optional LangChain layer.

The same six-stage pipeline powers:
  • Ad auctions         (competitive bidding, eCPM maximisation)
  • Health insurance    (plan–patient matching, enrollment prediction)
  • E-commerce search   (product ranking, purchase likelihood)
  • Content rec.        (Netflix/Spotify, watch/listen probability)
  • Job matching        (candidate–role fit scoring)

Change the data. Keep the engine.

Sections:
  1A. Data — AdTech domain
  1B. Data — Health Insurance domain
  2.  Retrieval      — inverted index + TF-IDF (the sparse baseline)
  3.  LinearRanker   — logistic regression, online SGD, BCE loss
  4.  NeuralRanker   — 2-layer MLP, hand-rolled forward + backprop
  5A. Auction        — second-price (Vickrey), for competitive markets
  5B. Ranking        — top-K, for recommendation without bidding
  6.  Pacing         — token bucket per supplier (budget smoothing)
  7.  FrequencyCap   — max impressions per user per supplier
  8.  Throttler      — adaptive bid gate (target win rate)
  9.  RAG Enrichment — LangChain + FAISS + open-source LLMs (Gemma2/Llama3/Mistral)
  10. Explainer      — chain-of-thought recommendation explanation
  11. ShadowScorer   — A/B test linear vs neural online (MLOps)
  12. Simulation A   — AdTech: auction + traffic shaping + online learning
  13. Simulation B   — Health Insurance: recommendation + CoT explanation
  14. Simulation C   — RAG enrichment pipeline (with LangChain fallback)

Inspired by Karpathy's microGPT (200-line GPT; this is the adtech sibling).
Production implementation (Go, BM25 + Qdrant, 250+ QPS):
  https://github.com/zizimind/openrtb-semantic-matcher
"""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────
# 1A. DATA — ADTECH
#     `value` = bid / max_bid  (normalised commercial value, 0-1)
# ─────────────────────────────────────────────────────────────

ADS = [
    {"id":"a01","title":"Nike Air Max Running Shoes",       "kw":["shoes","running","athletic","sport","nike"],         "bid":2.50,"value":0.50,"budget":100.0,"advertiser":"nike",    "quality":0.85},
    {"id":"a02","title":"Adidas Ultraboost Trail Running",  "kw":["shoes","trail","running","adidas","outdoor"],        "bid":2.20,"value":0.44,"budget": 80.0,"advertiser":"adidas",  "quality":0.80},
    {"id":"a03","title":"MacBook Pro M3",                   "kw":["laptop","macbook","apple","computer","pro"],         "bid":5.00,"value":1.00,"budget":200.0,"advertiser":"apple",   "quality":0.90},
    {"id":"a04","title":"Dell XPS 15 Premium Laptop",       "kw":["laptop","dell","windows","computer","xps"],          "bid":4.50,"value":0.90,"budget":150.0,"advertiser":"dell",    "quality":0.75},
    {"id":"a05","title":"Spotify Premium",                  "kw":["music","streaming","spotify","premium","audio"],     "bid":1.80,"value":0.36,"budget": 60.0,"advertiser":"spotify", "quality":0.70},
    {"id":"a06","title":"Netflix — Watch Anywhere",         "kw":["streaming","movies","netflix","video","tv"],         "bid":2.00,"value":0.40,"budget":120.0,"advertiser":"netflix", "quality":0.78},
    {"id":"a07","title":"Allbirds Wool Runners",            "kw":["shoes","wool","sustainable","running","allbirds"],   "bid":1.50,"value":0.30,"budget": 40.0,"advertiser":"allbirds","quality":0.65},
    {"id":"a08","title":"Amazon Echo Dot",                  "kw":["smart","speaker","amazon","echo","alexa"],           "bid":3.00,"value":0.60,"budget": 90.0,"advertiser":"amazon",  "quality":0.82},
    {"id":"a09","title":"Google Pixel 8 Pro",               "kw":["phone","smartphone","google","pixel","android"],     "bid":4.00,"value":0.80,"budget":180.0,"advertiser":"google",  "quality":0.88},
    {"id":"a10","title":"Samsung Galaxy S24 Ultra",         "kw":["phone","samsung","galaxy","android","smartphone"],   "bid":3.80,"value":0.76,"budget":160.0,"advertiser":"samsung", "quality":0.83},
]

AD_USERS = [
    {"id":"u1","interests":["running","sport","outdoor"],         "segment":"fitness"},
    {"id":"u2","interests":["laptop","computer","tech"],          "segment":"professional"},
    {"id":"u3","interests":["music","streaming","entertainment"],  "segment":"consumer"},
    {"id":"u4","interests":["phone","smartphone","tech"],         "segment":"mobile"},
    {"id":"u5","interests":["shoes","fashion","lifestyle"],       "segment":"lifestyle"},
]

AD_QUERIES = [
    ["running","shoes","sport"], ["laptop","computer","pro"],
    ["music","streaming"],       ["phone","smartphone"],
    ["shoes","outdoor"],         ["smart","speaker"],
    ["streaming","video"],
]


# ─────────────────────────────────────────────────────────────
# 1B. DATA — HEALTH INSURANCE
#     `value` = monthly_premium / max_premium  (insurer revenue, normalised)
#     No `bid` field — no auction, pure recommendation.
# ─────────────────────────────────────────────────────────────

HEALTH_PLANS = [
    {"id":"h01","title":"BlueCross Silver PPO",     "kw":["ppo","silver","flexible","specialist","nationwide"],    "value":0.62,"budget":500.0,"advertiser":"bluecross","quality":0.88},
    {"id":"h02","title":"Kaiser Gold HMO",          "kw":["hmo","gold","preventive","integrated","affordable"],   "value":0.56,"budget":500.0,"advertiser":"kaiser",   "quality":0.85},
    {"id":"h03","title":"Aetna Bronze HSA",         "kw":["hsa","bronze","low-cost","young","high-deductible"],   "value":0.31,"budget":500.0,"advertiser":"aetna",    "quality":0.72},
    {"id":"h04","title":"UnitedHealth Platinum PPO","kw":["ppo","platinum","comprehensive","family","premium"],   "value":1.00,"budget":500.0,"advertiser":"united",   "quality":0.91},
    {"id":"h05","title":"Cigna Silver HMO",         "kw":["hmo","silver","mental-health","telehealth","digital"], "value":0.51,"budget":500.0,"advertiser":"cigna",    "quality":0.79},
    {"id":"h06","title":"Humana Bronze HMO",        "kw":["hmo","bronze","low-cost","basic","young"],             "value":0.28,"budget":500.0,"advertiser":"humana",   "quality":0.68},
]

PATIENTS = [
    {"id":"p1","interests":["preventive","affordable","young"],      "profile":"young_healthy"},
    {"id":"p2","interests":["specialist","flexible","nationwide"],   "profile":"mid_career"},
    {"id":"p3","interests":["family","comprehensive","premium"],     "profile":"family_plan"},
    {"id":"p4","interests":["mental-health","telehealth","digital"], "profile":"tech_savvy"},
    {"id":"p5","interests":["hsa","low-cost","high-deductible"],     "profile":"budget_conscious"},
]

PATIENT_QUERIES = [
    ["affordable","preventive","young"], ["specialist","flexible","ppo"],
    ["family","comprehensive"],          ["mental-health","telehealth"],
    ["hsa","low-cost"],                  ["integrated","gold"],
]


# ─────────────────────────────────────────────────────────────
# 2. RETRIEVAL — Inverted Index + TF-IDF
#    Domain-agnostic: works for ads, health plans, products, jobs.
# ─────────────────────────────────────────────────────────────

class ItemIndex:
    """
    Sparse retrieval over item keywords.

    TF  = term_count / doc_length
    IDF = log(N / df(t))           ← rare terms score higher
    Score(item, query) = Σ TF(t) × IDF(t)   for t in query ∩ item.kw

    Production upgrade path:
      • Stage 1 (this):  TF-IDF, exact keyword match
      • Stage 2:         BM25 (saturates TF, adds doc-length norm)
      • Stage 3:         Dense vectors (SBERT / text-embedding-3-small)
      • Stage 4:         Hybrid BM25 + dense via Reciprocal Rank Fusion
      • Stage 5:         LLM cross-encoder re-ranker on top-50 candidates
    """

    def __init__(self, items: Sequence[Dict[str, Any]]):
        self.items = {it["id"]: it for it in items}
        self.N = len(items)
        self.inv = defaultdict(list)
        self.df = defaultdict(int)
        for it in items:
            seen = set()
            for term in it["kw"]:
                self.inv[term].append(it["id"])
                if term not in seen:
                    self.df[term] += 1
                    seen.add(term)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(self.N / df) if df else 0.0

    def search(
        self,
        query_terms: Sequence[str],
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if top_k <= 0:
            return []
        scores = defaultdict(float)
        for term in query_terms:
            idf = self._idf(term)
            for iid in self.inv.get(term, []):
                it = self.items[iid]
                tf = it["kw"].count(term) / len(it["kw"])
                scores[iid] += tf * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.items[iid], score) for iid, score in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────
# 3. LINEAR RANKER  (logistic regression, online SGD)
# ─────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(u * v for u, v in zip(a, b))


class LinearRanker:
    """
    Predicts the probability of a positive outcome (click / enroll / purchase).

      p_hat = sigmoid(w · x + b)

    Feature vector x = [semantic_sim, value_norm, quality, user_affinity]
      semantic_sim  — TF-IDF score (how well item matches query)
      value_norm    — item["value"] (bid/max_bid for ads; premium/max for plans)
      quality       — item["quality"] (historical CTR, plan rating, etc.)
      user_affinity — fraction of item keywords in user interests

    These four features are IDENTICAL across all domains.
    Only the label changes: click (adtech), enroll (health), purchase (ecomm).

    Loss:     Binary Cross-Entropy
    Gradient: dL/dz = p_hat − label          (BCE + sigmoid, chain rule)
    Update:   w ← w − lr × grad × x          (online SGD, one sample)
    """

    def __init__(self, lr=0.05):
        # No global seed here — caller controls reproducibility via random.seed().
        # Seeding inside a constructor mutates global state and breaks any code
        # that creates a LinearRanker mid-simulation.
        self.w  = [random.gauss(0, 0.1) for _ in range(4)]
        self.b  = 0.0
        self.lr = lr

    def features(self, item, tfidf_score, user):
        semantic  = min(tfidf_score, 1.0)
        value     = item["value"]                            # already normalised
        quality   = item["quality"]
        affinity  = sum(1 for kw in item["kw"] if kw in user["interests"]) \
                    / max(len(user["interests"]), 1)
        return [semantic, value, quality, affinity]

    def predict(self, item, tfidf_score, user):
        """p(click | item, query, user)  — works for any domain."""
        x = self.features(item, tfidf_score, user)
        return sigmoid(dot(self.w, x) + self.b)

    def ecpm(self, item, tfidf_score, user):
        """AdTech: effective cost-per-mille = p_hat × bid × 1000."""
        return self.predict(item, tfidf_score, user) * item.get("bid", 1.0) * 1000.0

    def fit_score(self, item, tfidf_score, user):
        """Recommendation: p_hat × quality × 100  (0–100 fit score)."""
        return self.predict(item, tfidf_score, user) * item["quality"] * 100.0

    def update(self, item, tfidf_score, user, label):
        x     = self.features(item, tfidf_score, user)
        p_hat = sigmoid(dot(self.w, x) + self.b)
        grad  = p_hat - label                                # dBCE/dz
        self.w = [wi - self.lr * grad * xi for wi, xi in zip(self.w, x)]
        self.b -= self.lr * grad


# ─────────────────────────────────────────────────────────────
# 4. NEURAL RANKER  (2-layer MLP, hand-rolled backprop)
# ─────────────────────────────────────────────────────────────

class NeuralRanker:
    """
    Architecture:  4 → [hidden=8, ReLU] → [1, Sigmoid]

    Forward pass:
      z1 = W1 @ x + b1      shape [H]    (linear transform)
      h  = relu(z1)         shape [H]    (non-linearity)
      z2 = W2 · h  + b2     scalar       (output logit)
      y  = sigmoid(z2)      scalar       (predicted probability)

    Backward pass — chain rule, spelled out:
      dL/dz2       = y − label           (BCE loss gradient at output)
      dL/dW2[j]    = dL/dz2 · h[j]
      dL/dh[j]     = dL/dz2 · W2[j]     (backprop through output layer)
      dL/dz1[i]    = dL/dh[i] · 1(z1>0) (backprop through ReLU)
      dL/dW1[i][j] = dL/dz1[i] · x[j]   (outer product)

    Same math as microGPT's autograd Value class — just unrolled for one network.
    In production: replace with PyTorch autograd + nn.Module.
    """

    def __init__(self, in_dim=4, hidden=8, lr=0.01):
        self.lr = lr
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = [[random.gauss(0, s1) for _ in range(in_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [random.gauss(0, s2) for _ in range(hidden)]
        self.b2 = 0.0

    @staticmethod
    def _relu(v):   return [max(0.0, x) for x in v]
    @staticmethod
    def _relu_d(v): return [1.0 if x > 0 else 0.0 for x in v]

    def forward(self, x):
        z1 = [dot(self.W1[i], x) + self.b1[i] for i in range(len(self.W1))]
        h  = self._relu(z1)
        z2 = dot(self.W2, h) + self.b2
        return sigmoid(z2), h, z1

    def predict(self, x):
        y, _, _ = self.forward(x)
        return y

    def backward(self, x, label):
        y, h, z1 = self.forward(x)
        dz2 = y - label
        dW2 = [dz2 * hi for hi in h]
        dh        = [dz2 * self.W2[j] for j in range(len(self.W2))]
        relu_mask = self._relu_d(z1)
        dz1       = [dh[i] * relu_mask[i] for i in range(len(z1))]
        dW1 = [[dz1[i] * x[j] for j in range(len(x))] for i in range(len(dz1))]
        self.W2 = [w - self.lr * g for w, g in zip(self.W2, dW2)]
        self.b2 -= self.lr * dz2
        self.W1 = [[self.W1[i][j] - self.lr * dW1[i][j] for j in range(len(x))]
                   for i in range(len(self.W1))]
        self.b1 = [b - self.lr * g for b, g in zip(self.b1, dz1)]
        return y


# ─────────────────────────────────────────────────────────────
# 5A. AUCTION — Second-Price (Vickrey)
#     Use when suppliers compete for a single slot (adtech, sponsored search).
# ─────────────────────────────────────────────────────────────

def run_auction(
    candidates: Sequence[Tuple[Dict[str, Any], float, float]],
    reserve_cpm: float = 500.0,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Second-price auction: winner pays second-highest eCPM + $0.001.

    Why second-price?  Truthful bidding is the dominant strategy.
    Proof sketch (Vickrey 1961):
      • Overbid → can win but pays more than item is worth → loss.
      • Underbid → can lose a profitable impression → missed revenue.
      • Bid true value → payment is always set by someone else → optimal.

    In Google Ads this is called "generalised second price" (GSP).
    OpenRTB 2.x implements it as AT=2 (auction type = second price).

    candidates: list of (item, tfidf_score, ecpm_float)
    returns:    (winner_item, clearing_price_$)  or  (None, 0)
    """
    if reserve_cpm < 0:
        raise ValueError("reserve_cpm must be non-negative")
    eligible = [(it, sc, e) for it, sc, e in candidates if e >= reserve_cpm]
    if not eligible:
        return None, 0.0
    eligible.sort(key=lambda x: x[2], reverse=True)
    winner    = eligible[0][0]
    floor_cpm = eligible[1][2] if len(eligible) > 1 else reserve_cpm
    return winner, floor_cpm / 1000.0 + 0.001


# ─────────────────────────────────────────────────────────────
# 5B. RANKING — Top-K Recommendation
#     Use when there is no auction: health plans, Netflix, job boards.
# ─────────────────────────────────────────────────────────────

def recommend_topk(
    candidates: Sequence[Tuple[Dict[str, Any], float, float]],
    k: int = 3,
) -> List[Tuple[Dict[str, Any], float, float]]:
    """
    Pure ranking: return the k items with the highest fit score.
    No payment, no bidding — just maximise relevance for the user.

    Same retrieval + scoring pipeline as the auction path.
    The only difference: the DECISION layer (rank vs. auction).

    This is the core insight: adtech IS a recommendation system
    with a price attached.  Remove the price → Netflix.
    Add a bid → Google Ads.
    """
    if k <= 0:
        return []
    eligible = sorted(candidates, key=lambda x: x[2], reverse=True)
    return eligible[:k]


# ─────────────────────────────────────────────────────────────
# 6. PACING — Token Bucket
# ─────────────────────────────────────────────────────────────

class TokenBucket:
    """
    Smooth budget delivery over a simulated day.

    Tokens refill at  rate = daily_budget / day_seconds.
    Each bid attempt consumes 1 token.
    Empty bucket → bid suppressed (paced out).

    Without pacing: a $100/day budget exhausts in the first hour,
    leaving 23 h with zero delivery and angry advertisers.
    The token bucket enforces smooth spend at the cost of a simple
    rate-limiting gate — O(1) per request.

    In production (openrtb-semantic-matcher): per-advertiser budget
    state lives in Redis; NATS JetStream propagates updates in <1 ms.
    """

    def __init__(self, daily_budget: float, day_seconds: float = 86_400):
        if daily_budget < 0:
            raise ValueError("daily_budget must be non-negative")
        if day_seconds <= 0:
            raise ValueError("day_seconds must be positive")
        self.rate = daily_budget / day_seconds
        self.capacity = daily_budget * 0.02  # 2% burst allowance
        self.tokens = self.capacity
        self._sim_t = 0.0

    def tick(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self.tokens = min(self.capacity, self.tokens + elapsed_seconds * self.rate)
        self._sim_t += elapsed_seconds

    def consume(self, cost: float = 1.0) -> bool:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


# ─────────────────────────────────────────────────────────────
# 7. FREQUENCY CAP
# ─────────────────────────────────────────────────────────────

class FrequencyCap:
    """
    Limit impressions per (user, supplier) pair per session.

    Adtech: prevents banner blindness, protects user experience.
    Health: prevents showing the same insurer too many times.
    Content: prevents the same artist dominating a playlist.

    Real systems: Redis key  user:{uid}:adv:{adv_id}  TTL=86400s.
    """

    def __init__(self, max_per_session=2):
        self.cap    = max_per_session
        self.counts = defaultdict(int)

    def allowed(self, user_id: str, supplier_id: str) -> bool:
        return self.counts[(user_id, supplier_id)] < self.cap

    def record(self, user_id: str, supplier_id: str) -> None:
        self.counts[(user_id, supplier_id)] += 1


# ─────────────────────────────────────────────────────────────
# 8. THROTTLER — Adaptive Bid Gate
# ─────────────────────────────────────────────────────────────

class Throttler:
    """
    Skip impressions probabilistically to approach a target win rate.

      throttle_prob = min(1, target_win_rate / observed_win_rate)

    Business case: in a real DSP, entering every auction costs money
    (network RTT, scoring compute, OpenRTB fees).  Throttling lets you
    concentrate spend on impressions where you have an edge.
    """

    def __init__(self, target_win_rate=0.20, warmup=10):
        self.target   = target_win_rate
        self.warmup   = warmup
        self.wins     = 0
        self.attempts = 0

    def should_bid(self):
        if self.attempts < self.warmup:
            return True
        observed = self.wins / self.attempts
        prob     = min(1.0, self.target / observed) if observed > 0 else 1.0
        return random.random() < prob

    def record(self, won: bool) -> None:
        self.attempts += 1
        if won:
            self.wins += 1


# ─────────────────────────────────────────────────────────────
# 9. RAG ENRICHMENT PIPELINE
#    LangChain + FAISS + open-source LLMs via Ollama
#
# The Semantic Matching Ladder (upgrade path for retrieval):
#   Stage 1 — TF-IDF/BM25 (this file's ItemIndex)  exact keyword match
#   Stage 2 — Dense vectors (SBERT / all-MiniLM)    semantic similarity
#   Stage 3 — Hybrid BM25 + dense via RRF           best of both worlds
#   Stage 4 — LLM re-ranker on top-50 candidates    chain-of-thought reasoning
#   Stage 5 — RAG enrichment (this section)         retrieve → generate → index
#
# Production: openrtb-semantic-matcher uses Stage 3 (BM25 + Qdrant + RRF).
#             DSPy + Ollama handles Stage 5 offline enrichment.
# ─────────────────────────────────────────────────────────────

# ── Supported open-source LLMs (run locally via Ollama) ──────
#
#   ollama pull <model>  then pass model_name= to RAGEnricher
#
SUPPORTED_LLMS = {
    # Google — "compressions": Gemma 2 is knowledge-distilled from Gemini
    "gemma2":       {"org": "Google",      "params": "9B",   "ram": "6 GB",  "notes": "best quality/size; default"},
    "gemma2:2b":    {"org": "Google",      "params": "2B",   "ram": "2 GB",  "notes": "fits on any laptop, fast"},
    # Meta
    "llama3":       {"org": "Meta",        "params": "8B",   "ram": "5 GB",  "notes": "strong general reasoning"},
    "llama3.1":     {"org": "Meta",        "params": "8B",   "ram": "5 GB",  "notes": "improved instruction follow"},
    # Mistral AI
    "mistral":      {"org": "Mistral AI",  "params": "7B",   "ram": "5 GB",  "notes": "excellent for RAG + retrieval"},
    "mixtral":      {"org": "Mistral AI",  "params": "8×7B", "ram": "26 GB", "notes": "MoE, near-GPT4 quality"},
    # Microsoft
    "phi3":         {"org": "Microsoft",   "params": "3.8B", "ram": "3 GB",  "notes": "tiny + fast, good for CoT"},
    "phi3:medium":  {"org": "Microsoft",   "params": "14B",  "ram": "9 GB",  "notes": "best Phi quality"},
    # Alibaba
    "qwen2":        {"org": "Alibaba",     "params": "7B",   "ram": "5 GB",  "notes": "multilingual, long context"},
    # Multimodal (image ads)
    "llava":        {"org": "LLaVA",       "params": "7B",   "ram": "5 GB",  "notes": "image+text, for visual ads"},
}

# ── Model Compression — two distinct layers ──────────────────
#
# Layer 1: WEIGHT quantisation (shrinks the model file)
#   Applied once before serving. Ollama handles this automatically via GGUF.
#
# Format      | Origin       | What it compresses   | Notes
# ────────────┼──────────────┼──────────────────────┼───────────────────────────
# GGUF Q4_K_M | llama.cpp    | Model weights 16→4bit | Default in Ollama; CPU-friendly
# GPTQ 4-bit  | community    | Model weights 16→4bit | GPU only; slightly better quality
# AWQ 4-bit   | MIT          | Model weights 16→4bit | Activation-aware; best accuracy/size
# BitsAndBytes| HuggingFace  | Model weights 8/4bit  | Dynamic; easiest to use in Python
#
# Layer 2: KV-CACHE quantisation (speeds up each inference call)
#   Applied at runtime. Reduces memory needed during generation.
#
# Format      | Origin       | What it compresses   | Notes
# ────────────┼──────────────┼──────────────────────┼───────────────────────────
# TurboQuant  | Google DM    | KV cache 16→3bit      | ICLR 2026; 6× memory, 8× faster
#             |              |                       | on H100; no accuracy loss
# FlexGen     | Stanford     | KV cache + weights    | CPU offloading for large models
#
# TurboQuant ≠ GGUF: they target DIFFERENT things and can be used together.
#   GGUF    → smaller model on disk / in RAM
#   TurboQuant → faster generation for long contexts (ads with many candidates)
#
# Gemma 2 architecture advantage: knowledge-distilled from Gemini Pro,
# so Gemma 2 9B outperforms Llama 3 8B on most benchmarks despite same size.
#
# Quick start:
#   ollama pull gemma2:2b   # 1.6 GB GGUF Q4 — runs on any laptop
#   ollama pull gemma2      # 5.4 GB GGUF Q4 — best open-source at 9B
#   ollama pull llama3      # 4.7 GB GGUF Q4
#   pip install langchain-huggingface langchain-ollama faiss-cpu sentence-transformers

try:
    from langchain_ollama import OllamaLLM
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    _LC = True
except ImportError:
    _LC = False

_ENRICH_TEMPLATE = """\
You are an expert in search relevance and recommendation systems.

Similar items already in the catalog:
{context}

New item to enrich:
  Title   : {title}
  Keywords: {keywords}
  Domain  : {domain}

User profile:
  Interests: {interests}
  Segment  : {segment}

Think step by step:
1. What user intents does this item satisfy beyond its literal keywords?
2. What semantic tags best capture its meaning (synonyms, categories, use-cases)?
3. Why would this specific user engage with this item?

Respond ONLY in this format:
INTENTS: <comma-separated 3-5 user intents>
TAGS: <comma-separated 5-8 semantic keywords>
REASON: <one sentence explaining the fit>
"""


class RAGEnricher:
    """
    Enriches items with LLM-generated semantic metadata before indexing.

    Pipeline (RAG = Retrieve → Augment → Generate):
      ┌─────────────────────────────────────────────────────────┐
      │  Item corpus ──► FAISS vector store (all-MiniLM-L6-v2) │
      │                          │                              │
      │  Query item ──► retrieve k=3 similar items (context)   │
      │                          │                              │
      │  CoT prompt ──► LLM (Gemma2 / Llama3 / Mistral)        │
      │                          │                              │
      │  Output: INTENTS + TAGS + REASON                        │
      │  → expand item keywords → better retrieval + ranking    │
      └─────────────────────────────────────────────────────────┘

    This is what openrtb-semantic-matcher does offline with DSPy + Ollama.
    LangChain replaces DSPy here for simpler orchestration and LLM swapping.

    Without LangChain installed: falls back to rule-based extraction.
    Same interface either way — the simulation runs in both modes.
    """

    def __init__(self, items, model_name="gemma2", domain="adtech"):
        self.domain     = domain
        self._llm       = None
        self._retriever = None
        self._prompt    = None

        if _LC:
            docs = [
                Document(
                    page_content=f"{it['title']}. Tags: {', '.join(it['kw'])}.",
                    metadata={"id": it["id"]}
                )
                for it in items
            ]
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",   # 80 MB, runs on CPU, no GPU needed
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            store           = FAISS.from_documents(docs, embeddings)
            self._retriever = store.as_retriever(search_kwargs={"k": 3})
            self._llm       = OllamaLLM(model=model_name, temperature=0.2)
            self._prompt    = PromptTemplate.from_template(_ENRICH_TEMPLATE)
            # LCEL chain: prompt | llm | output parser
            self._chain     = self._prompt | self._llm | StrOutputParser()

    def enrich(self, item, user):
        """
        Returns {"intents": [...], "tags": [...], "reason": "..."}.
        Calls real LLM when LangChain available, rule-based otherwise.
        """
        if not _LC or self._llm is None:
            return self._rule_based(item, user)
        try:
            # Step 1 — retrieve similar items as grounding context
            similar = self._retriever.invoke(item["title"])
            context = "\n".join(f"  • {d.page_content}" for d in similar)
            # Step 2 — invoke LCEL chain (prompt → LLM → parse)
            raw = self._chain.invoke({
                "context":   context,
                "title":     item["title"],
                "keywords":  ", ".join(item["kw"]),
                "domain":    self.domain,
                "interests": ", ".join(user["interests"]),
                "segment":   user.get("segment", user.get("profile", "general")),
            })
            return self._parse(raw)
        except Exception:
            return self._rule_based(item, user)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        result = {"intents": [], "tags": [], "reason": ""}
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("INTENTS:"):
                result["intents"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
            elif line.upper().startswith("TAGS:"):
                result["tags"]    = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
            elif line.upper().startswith("REASON:"):
                result["reason"]  = line.split(":", 1)[1].strip()
        if not result["reason"]:
            result["reason"] = "No explicit reason provided by model."
        return result

    @staticmethod
    def _rule_based(item, user):
        overlap = [kw for kw in item["kw"] if kw in user["interests"]]
        extra   = [item["title"].split()[0].lower(), item.get("advertiser",""), "recommended"]
        return {
            "intents": [f"find {kw}" for kw in overlap[:3]] or ["general discovery"],
            "tags":    list(set(item["kw"] + [e for e in extra if e]))[:8],
            "reason":  f"keyword overlap on [{', '.join(overlap)}]" if overlap
                       else f"quality={item['quality']:.2f} broad match",
        }


# ─────────────────────────────────────────────────────────────
# 10. EXPLAINER — Chain-of-Thought Recommendation
# ─────────────────────────────────────────────────────────────

def explain_recommendation(item, user, score):
    """
    Generate a human-readable explanation for a recommendation.

    In production this is where an LLM lives:
      prompt = f'''
        User profile: {user["profile"]}, interests: {user["interests"]}
        Recommended item: {item["title"]}
        Fit score: {score:.1f}/100
        Think step by step: why is this item a good fit?
        Answer in one sentence.
      '''
      explanation = llm(prompt, temperature=0.3)

    Here: rule-based template to show the pattern without a real model.
    The structure (retrieve evidence → reason → conclude) mirrors CoT.
    """
    matching = [kw for kw in item["kw"] if kw in user["interests"]]
    parts = []
    if matching:
        parts.append(f"matches your interest in {', '.join(matching)}")
    if item["quality"] >= 0.85:
        parts.append("highly rated plan/product")
    if score >= 70:
        parts.append("strong overall fit")
    elif score >= 50:
        parts.append("good fit")
    else:
        parts.append("partial fit — consider reviewing details")
    return "Fit {:.0f}/100 — {}.".format(score, "; ".join(parts))


# ─────────────────────────────────────────────────────────────
# 11. SHADOW SCORER — A/B Test Linear vs Neural  (MLOps)
# ─────────────────────────────────────────────────────────────

class ShadowScorer:
    """
    Run two rankers in parallel; serve one, log both.
    This is the standard MLOps pattern for safe model upgrades:

      champion  = LinearRanker  (live, controls traffic)
      challenger = NeuralRanker  (shadow, scores but does not serve)

    After N requests: compare champion vs. challenger on the same labels.
    If challenger has lower log-loss → promote it.
    If not → keep champion, debug challenger.

    In production: shadow scores are logged to Kafka / BigQuery.
    A Spark/dbt job computes offline metrics nightly.
    Online A/B test (10% traffic split) follows to measure business metrics.
    """

    def __init__(self):
        self.champion   = LinearRanker(lr=0.05)
        self.challenger = NeuralRanker(lr=0.01)
        self._champ_loss = 0.0
        self._chal_loss  = 0.0
        self._n          = 0

    def score(self, item, tfidf, user, use_ecpm=True):
        x     = self.champion.features(item, tfidf, user)
        p_c   = self.champion.predict(item, tfidf, user)
        p_n   = self.challenger.predict(x)
        if use_ecpm:
            return p_c * item.get("bid", 1.0) * 1000.0, p_n * item.get("bid", 1.0) * 1000.0
        return p_c * item["quality"] * 100.0, p_n * item["quality"] * 100.0

    def update(self, item, tfidf, user, label):
        x = self.champion.features(item, tfidf, user)
        # Measure loss BEFORE updating — this reflects actual prediction quality,
        # not the rosier post-update picture.
        p_c = self.champion.predict(item, tfidf, user)
        p_n = self.challenger.predict(x)
        eps = 1e-9
        self._champ_loss += -(label * math.log(p_c + eps) + (1-label) * math.log(1-p_c + eps))
        self._chal_loss  += -(label * math.log(p_n + eps) + (1-label) * math.log(1-p_n + eps))
        self._n          += 1
        # Then apply gradient updates.
        self.champion.update(item, tfidf, user, label)
        self.challenger.backward(x, label)

    def report(self):
        if self._n == 0:
            return
        print(f"\n  Shadow Scoring Report  (n={self._n})")
        print(f"  Champion  (linear) avg log-loss : {self._champ_loss/self._n:.4f}")
        print(f"  Challenger (neural) avg log-loss: {self._chal_loss/self._n:.4f}")
        winner = "challenger → promote" if self._chal_loss < self._champ_loss else "champion → keep"
        print(f"  Decision: {winner}")


# ─────────────────────────────────────────────────────────────
# 12. SIMULATION A — AdTech
# ─────────────────────────────────────────────────────────────

def simulate_adtech(n_requests=60, seed=1):
    random.seed(seed)
    print(f"\n{'═'*67}")
    print(f"  Simulation A — AdTech Auction  |  {n_requests} requests")
    print(f"{'═'*67}")

    index   = ItemIndex(ADS)
    shadow  = ShadowScorer()
    SIM_DAY = 3_600.0
    dt      = SIM_DAY / n_requests
    buckets = {ad["advertiser"]: TokenBucket(ad["budget"], SIM_DAY) for ad in ADS}
    fcap    = FrequencyCap(max_per_session=2)
    throttler = Throttler(target_win_rate=0.20)

    spent = defaultdict(float)
    wins  = defaultdict(int)
    total = 0.0
    nofill = 0

    print(f"\n  {'#':>3}  {'user':<4}  {'query':<18}  {'winner':<10}  {'adv':<10}  {'$price':>7}  click")
    print(f"  {'─'*3}  {'─'*4}  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*5}")

    for i in range(n_requests):
        user  = random.choice(AD_USERS)
        query = random.choice(AD_QUERIES)

        for b in buckets.values():
            b.tick(dt)

        raw = index.search(query, top_k=5)
        if not raw:
            nofill += 1; continue

        # Throttler is a per-request gate: should we participate in this auction?
        # Tracks wins/attempts across all requests; skips low-value inventory.
        if not throttler.should_bid():
            nofill += 1; continue

        candidates = []
        for item, tfidf in raw:
            adv = item["advertiser"]
            if not buckets[adv].consume():        continue
            if not fcap.allowed(user["id"], adv): continue
            champ_ecpm, _ = shadow.score(item, tfidf, user, use_ecpm=True)
            candidates.append((item, tfidf, champ_ecpm))

        if not candidates:
            throttler.record(won=False)
            nofill += 1; continue

        winner, price = run_auction(candidates, reserve_cpm=500.0)
        if winner is None:
            throttler.record(won=False)
            nofill += 1; continue

        clicked = random.random() < winner["quality"] * 0.30
        label   = 1.0 if clicked else 0.0
        wt      = next(sc for it, sc, _ in candidates if it["id"] == winner["id"])
        shadow.update(winner, wt, user, label)

        fcap.record(user["id"], winner["advertiser"])
        throttler.record(won=True)
        spent[winner["advertiser"]] += price
        wins[winner["advertiser"]]  += 1
        total += price

        if i < 12 or i % 12 == 0:
            q = query[0] + "+" + query[-1]
            print(f"  {i+1:>3}  {user['id']:<4}  {q:<18}  {winner['id']:<10}  "
                  f"{winner['advertiser']:<10}  ${price:>6.3f}  {'✓' if clicked else '·'}")

    print(f"\n  {'─'*67}")
    print(f"  Revenue: ${total:.2f}  |  No-fill: {nofill}/{n_requests}")
    print(f"\n  {'Advertiser':<12} {'Wins':>5}  {'Spent':>8}  {'$/win':>7}")
    print(f"  {'─'*12} {'─'*5}  {'─'*8}  {'─'*7}")
    for adv in sorted(wins, key=lambda a: wins[a], reverse=True):
        cpp = spent[adv] / wins[adv]
        print(f"  {adv:<12} {wins[adv]:>5}  ${spent[adv]:>7.2f}  ${cpp:>6.3f}")

    shadow.report()
    print(f"\n  Linear weights: {['%+.3f'%w for w in shadow.champion.w]}")
    print(f"  features: [semantic_sim, value_norm, quality, user_affinity]")


# ─────────────────────────────────────────────────────────────
# 13. SIMULATION B — Health Insurance Recommendation
# ─────────────────────────────────────────────────────────────

def simulate_health(n_requests=15, seed=2):
    random.seed(seed)
    print(f"\n{'═'*67}")
    print(f"  Simulation B — Health Plan Recommendation  |  {n_requests} patients")
    print(f"  Same pipeline as adtech. No auction — pure top-3 ranking.")
    print(f"{'═'*67}")

    index  = ItemIndex(HEALTH_PLANS)
    ranker = LinearRanker(lr=0.05)

    for i in range(n_requests):
        patient = random.choice(PATIENTS)
        query   = random.choice(PATIENT_QUERIES)

        raw = index.search(query, top_k=6)
        if not raw:
            continue

        # Score each plan by fit (no bid, no auction)
        candidates = []
        for plan, tfidf in raw:
            fs = ranker.fit_score(plan, tfidf, patient)
            candidates.append((plan, tfidf, fs))

        # Top-3 recommendation (no auction)
        top3 = recommend_topk(candidates, k=3)
        if not top3:
            continue

        if i < 8:
            q = query[0] + "+" + query[-1]
            print(f"\n  Patient {patient['id']} ({patient['profile']:<18}) | query: {q}")
            for rank, (plan, tfidf, fs) in enumerate(top3, 1):
                note = explain_recommendation(plan, patient, fs)
                print(f"    {rank}. [{plan['id']}] {plan['title']:<30}  score={fs:5.1f}  {note}")

        # Simulate enrollment: enroll with prob = quality × 0.4
        enrolled = random.random() < top3[0][0]["quality"] * 0.40
        ranker.update(top3[0][0], top3[0][1], patient, 1.0 if enrolled else 0.0)

    print(f"\n  Ranker weights after {n_requests} patients:")
    print(f"  w = {['%+.3f'%w for w in ranker.w]}  b = {ranker.b:+.3f}")
    print(f"  features: [semantic_sim, value_norm, quality, user_affinity]")
    print(f"\n  Note: same w[] structure as AdTech simulation.")
    print(f"  The ranker does not know which domain it is in.")


# ─────────────────────────────────────────────────────────────
# 14. SIMULATION C — RAG Enrichment Pipeline
# ─────────────────────────────────────────────────────────────

def simulate_rag(model_name="gemma2", seed=3):
    """
    Shows RAG enrichment for both AdTech and Health Insurance domains.

    With Ollama running locally:
      ollama pull gemma2:2b   (fastest, 2 GB)
      ollama pull gemma2      (best quality, 6 GB)
      ollama pull llama3      (alternative, 5 GB)

    Without Ollama: runs in rule-based fallback mode.
    Same interface, same output structure — just without the LLM reasoning.
    """
    random.seed(seed)
    mode = f"LangChain + {model_name}" if _LC else "rule-based fallback (pip install langchain-community faiss-cpu sentence-transformers)"
    print(f"\n{'═'*67}")
    print(f"  Simulation C — RAG Enrichment  |  {mode}")
    print(f"{'═'*67}")

    samples = [
        ("adtech",  ADS,          AD_USERS,  "AdTech"),
        ("health",  HEALTH_PLANS, PATIENTS,  "Health Insurance"),
    ]

    for domain, items, users, label in samples:
        print(f"\n  ── {label} domain ──")
        enricher = RAGEnricher(items, model_name=model_name, domain=domain)
        for item in random.sample(items, min(3, len(items))):
            user   = random.choice(users)
            result = enricher.enrich(item, user)
            seg    = user.get("segment", user.get("profile", ""))
            print(f"\n  Item    : {item['title']}")
            print(f"  User    : {seg} | interests: {', '.join(user['interests'][:3])}")
            print(f"  Intents : {', '.join(result['intents'])}")
            print(f"  Tags    : {', '.join(result['tags'])}")
            print(f"  Reason  : {result['reason']}")

    print(f"\n  ── LLM Catalog (ollama pull <model>) ──")
    print(f"  {'Model':<16} {'Org':<12} {'Params':<8} {'RAM':<7} Notes")
    print(f"  {'─'*16} {'─'*12} {'─'*8} {'─'*7} {'─'*30}")
    for model, info in SUPPORTED_LLMS.items():
        print(f"  {model:<16} {info['org']:<12} {info['params']:<8} {info['ram']:<7} {info['notes']}")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    simulate_adtech(n_requests=60, seed=1)
    simulate_health(n_requests=15, seed=2)
    simulate_rag(model_name="gemma2", seed=3)


if __name__ == "__main__":
    main()
