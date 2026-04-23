# MiniAdTech — System Design & Decision Workflows

**Architecture diagrams, data flows, and decision logic for every component.**

---

## 1. The Full Request Lifecycle (100ms window)

Every impression — ad, health plan, product — flows through the same six stages.

```
  USER / BROWSER
       │
       │  "running shoes" query, user_id=u1, page_context
       ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                        GATEWAY  (Go / FastAPI)                 │
  │  • parse OpenRTB / HTTP request                                │
  │  • attach user profile from feature store / cache             │
  │  • route to matching engine                                    │
  └─────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 1 — RETRIEVAL                          │
  │                                                                 │
  │   ItemIndex.search(query_terms, top_k=50)                       │
  │                                                                 │
  │   Sparse:  TF-IDF over inverted index       →  candidates       │
  │   Dense:   FAISS cosine similarity          →  merged           │
  │   Fusion:  Reciprocal Rank Fusion (RRF)     →  top-50           │
  └─────────────────────────┬───────────────────────────────────────┘
                            │  50 candidates
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 2 — TRAFFIC SHAPING                    │
  │                                                                 │
  │   for each candidate:                                           │
  │     ① Throttler.should_bid()    →  skip if win-rate too high    │
  │     ② TokenBucket.consume()     →  skip if budget paced out     │
  │     ③ FrequencyCap.allowed()    →  skip if user seen too much   │
  │                                                                 │
  │   Output: eligible candidates (typically 5–15)                  │
  └─────────────────────────┬───────────────────────────────────────┘
                            │  5–15 eligible
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 3 — SCORING  (ML)                      │
  │                                                                 │
  │   x = [semantic_sim, value_norm, quality, user_affinity]        │
  │                                                                 │
  │   LinearRanker:  p = sigmoid(w·x + b)                          │
  │   NeuralRanker:  p = MLP(x)  [shadow mode]                      │
  │                                                                 │
  │   AdTech score:   eCPM = p × bid × 1000                         │
  │   Health score:   fit  = p × quality × 100                      │
  └─────────────────────────┬───────────────────────────────────────┘
                            │  (item, tfidf, score) list
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 4 — DECISION                           │
  │                                                                 │
  │   Has auction? ─── YES ──► run_auction()  → 1 winner + price    │
  │                │                                                │
  │                NO ───────► recommend_topk() → ranked list       │
  └─────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 5 — FEEDBACK + LEARNING                │
  │                                                                 │
  │   User clicks / enrolls / purchases  →  label = 1              │
  │   User ignores                       →  label = 0              │
  │                                                                 │
  │   LinearRanker.update(item, tfidf, user, label)  ← online SGD  │
  │   NeuralRanker.backward(x, label)               ← backprop     │
  │   ShadowScorer logs both losses for comparison                  │
  └─────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STAGE 6 — SERVE + LOG                        │
  │                                                                 │
  │   Return: winner ad / top-3 plans / ranked products             │
  │   Log:    request_id, user_id, item_id, score, price, label     │
  │   Async:  Kafka → data warehouse → nightly model retraining     │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                        MINI ADTECH ENGINE                         │
  │                                                                   │
  │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐ │
  │  │  ITEM INDEX  │    │  ML RANKERS  │    │  TRAFFIC SHAPERS    │ │
  │  │              │    │              │    │                     │ │
  │  │  inverted    │    │  LinearRanker│    │  TokenBucket        │ │
  │  │  index       │───►│  (champion)  │    │  (pacing)           │ │
  │  │              │    │              │    │                     │ │
  │  │  TF-IDF      │    │  NeuralRanker│    │  FrequencyCap       │ │
  │  │  scoring     │    │  (challenger)│    │  (impressions)      │ │
  │  │              │    │              │    │                     │ │
  │  │  FAISS       │    │  ShadowScore │    │  Throttler          │ │
  │  │  (optional)  │    │  r (A/B)     │    │  (bid gate)         │ │
  │  └──────────────┘    └──────────────┘    └─────────────────────┘ │
  │          │                   │                      │             │
  │          └───────────────────┴──────────────────────┘             │
  │                              │                                    │
  │                    ┌─────────┴──────────┐                         │
  │                    │                    │                         │
  │             ┌──────▼──────┐    ┌────────▼───────┐                │
  │             │   AUCTION   │    │   RANKING      │                │
  │             │  (AdTech)   │    │  (Health/Rec.) │                │
  │             │             │    │                │                │
  │             │ Vickrey 2nd │    │ Top-K by score │                │
  │             │ price rule  │    │ + CoT explain  │                │
  │             └──────┬──────┘    └────────┬───────┘                │
  │                    └──────────┬──────────┘                        │
  │                               │                                   │
  │                    ┌──────────▼──────────┐                        │
  │                    │    RAG ENRICHER     │                        │
  │                    │  (pre-index step)   │                        │
  │                    │                     │                        │
  │                    │  FAISS vector store │                        │
  │                    │  LangChain LCEL     │                        │
  │                    │  Gemma2/Llama3/     │                        │
  │                    │  Mistral via Ollama │                        │
  │                    └─────────────────────┘                        │
  └───────────────────────────────────────────────────────────────────┘
```

---

## 3. Decision Flow — Auction vs Recommendation

```
                     Eligible candidates arrive
                               │
               ┌───────────────┴────────────────┐
               │                                │
        COMPETITIVE MARKET?               PURE RECOMMENDATION?
        (AdTech, Sponsored Search,        (Health Insurance,
         Job Board sponsored slots)        Netflix, Spotify,
               │                           E-commerce organic)
               │                                │
               ▼                                ▼
      ┌─────────────────┐              ┌─────────────────────┐
      │   run_auction() │              │  recommend_topk()   │
      │                 │              │                     │
      │  Sort by eCPM   │              │  Sort by fit_score  │
      │  = p × bid      │              │  = p × quality      │
      │    × 1000       │              │    × 100            │
      │                 │              │                     │
      │  ≥ reserve_cpm? │              │  Return top-K       │
      │  YES → winner   │              │  (no floor needed)  │
      │  NO  → no fill  │              │                     │
      └────────┬────────┘              └──────────┬──────────┘
               │                                  │
               ▼                                  ▼
      Payment = 2nd price              explain_recommendation()
      + $0.001 floor                   → CoT reason per item
               │                                  │
               └──────────────┬───────────────────┘
                              │
                    Serve result to user
                    Log outcome (click / enroll / skip)
                    Update model online (SGD step)
```

---

## 4. Traffic Shaping Decision Tree

Every candidate passes through three sequential gates before scoring.
**Any gate can eliminate a candidate.** All gates run in O(1).

```
  Candidate arrives (ad / plan / product)
          │
          ▼
  ┌───────────────────────────────┐
  │  GATE 1: THROTTLER            │
  │                               │
  │  observed_win_rate =          │
  │    wins / attempts            │
  │                               │
  │  throttle_prob =              │
  │    target_rate /              │
  │    observed_rate              │
  │                               │
  │  random() < throttle_prob?    │
  └────────┬──────────────────────┘
           │ YES                NO
           │              ──────────►  SKIP  (save compute cost)
           ▼
  ┌───────────────────────────────┐
  │  GATE 2: TOKEN BUCKET         │
  │                               │
  │  tokens refill at:            │
  │    rate = budget / day_secs   │
  │                               │
  │  burst cap = 2% of budget     │
  │                               │
  │  tokens >= 1.0?               │
  └────────┬──────────────────────┘
           │ YES                NO
           │              ──────────►  SKIP  (budget paced out)
           ▼
  ┌───────────────────────────────┐
  │  GATE 3: FREQUENCY CAP        │
  │                               │
  │  count[(user_id, supplier)]   │
  │  < max_per_session (2)?       │
  └────────┬──────────────────────┘
           │ YES                NO
           │              ──────────►  SKIP  (user protected)
           ▼
  ELIGIBLE — proceed to scoring
```

**Business logic behind each gate:**

| Gate | Protects | Failure cost without it |
|------|----------|------------------------|
| Throttler | System compute / DSP fees | Bidding on every worthless impression |
| Token Bucket | Advertiser daily budget | Budget gone by 9am, zero delivery rest of day |
| Frequency Cap | User experience | Banner blindness, brand damage, churn |

---

## 5. ML Model Architecture

```
  INPUT FEATURES  (same for all domains)
  ┌─────────────────────────────────────────────────────┐
  │  x[0]  semantic_sim   TF-IDF score (0–1)            │
  │  x[1]  value_norm     bid/max_bid or premium/max    │
  │  x[2]  quality        historical CTR / plan rating  │
  │  x[3]  user_affinity  keyword overlap with interests│
  └────────────────────────┬────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
  ┌────────▼──────────┐          ┌─────────▼──────────┐
  │  LINEAR RANKER    │          │  NEURAL RANKER     │
  │  (champion)       │          │  (challenger)      │
  │                   │          │                    │
  │  z = w·x + b      │          │  z1 = W1@x + b1    │
  │  p = sigmoid(z)   │          │  h  = relu(z1)     │
  │                   │          │  z2 = W2·h + b2    │
  │  4 weights        │          │  p  = sigmoid(z2)  │
  │  1 bias           │          │                    │
  │  5 params total   │          │  4×8 + 8 + 8×1 + 1 │
  │                   │          │  = 49 params       │
  └────────┬──────────┘          └─────────┬──────────┘
           │                               │
           │  p_champion                   │  p_challenger
           │   (serves)                    │   (shadow log)
           └───────────────┬───────────────┘
                           │
                  ┌────────▼────────┐
                  │  SHADOW SCORER  │
                  │                 │
                  │  both losses    │
                  │  tracked online │
                  │                 │
                  │  after N steps: │
                  │  lower loss     │
                  │  → promote      │
                  └─────────────────┘

  LOSS FUNCTION (both models):
  L = −[label × log(p) + (1−label) × log(1−p)]   (Binary Cross-Entropy)

  GRADIENT (online SGD, one sample at a time):
  dL/dz = p − label
  w ← w − lr × (p − label) × x
```

---

## 6. MLOps Decision Workflow — Safe Model Promotion

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  PHASE 1: SHADOW MODE  (current state)                           │
  │                                                                  │
  │  Every request:                                                  │
  │    Champion scores → serves the result                          │
  │    Challenger scores → logs only, never serves                  │
  │    Both receive the same feedback label                         │
  │                                                                  │
  │  After N=1000 requests:                                         │
  │    champion_loss  = avg BCE over all samples                    │
  │    challenger_loss = avg BCE over all samples                   │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────────┐        │
  │  │  challenger_loss < champion_loss?                    │        │
  │  └──────────┬─────────────────────────────┬────────────┘        │
  │           YES                            NO                     │
  │             │                             │                     │
  │             ▼                             ▼                     │
  │   PHASE 2: A/B TEST              Stay in shadow.               │
  │   10% live traffic               Debug challenger.             │
  │   to challenger                                                 │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  PHASE 2: A/B TEST  (online experiment)                          │
  │                                                                  │
  │  90% of requests → Champion                                     │
  │  10% of requests → Challenger                                   │
  │                                                                  │
  │  Measure BUSINESS metrics (not just ML loss):                   │
  │    AdTech:  Revenue per 1000 impressions (RPM)                  │
  │    Health:  Enrollment rate, plan fit score                     │
  │    Content: Watch time, completion rate                         │
  │                                                                  │
  │  Statistical significance reached? (p < 0.05, n > 10,000)      │
  │  ┌──────────────────────────────────────────────────────┐        │
  │  │  Challenger metrics better?                          │        │
  │  └──────────┬─────────────────────────────┬────────────┘        │
  │           YES                            NO                     │
  │             │                             │                     │
  │             ▼                             ▼                     │
  │   PHASE 3: PROMOTE               Rollback 100% to              │
  │   Swap champion ↔ challenger     champion. Retire               │
  │   Old champion → new shadow      challenger.                   │
  └──────────────────────────────────────────────────────────────────┘

  In code: ShadowScorer.report() runs Phase 1 automatically.
  Phases 2–3 require a traffic splitting layer (nginx / Istio / LaunchDarkly).
```

---

## 7. RAG Enrichment Pipeline

**When it runs:** offline, before items are indexed. Not on the hot path.

```
  ITEM CORPUS  (ads, health plans, products, jobs)
       │
       │  "Nike Air Max Running Shoes. Tags: shoes, running, athletic"
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1 — BUILD VECTOR STORE                                    │
  │                                                                 │
  │  HuggingFaceEmbeddings("all-MiniLM-L6-v2")                     │
  │    → 384-dim dense vector per item                              │
  │    → stored in FAISS index (L2 / cosine similarity)            │
  │                                                                 │
  │  80 MB model, runs on CPU, no GPU needed                       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  FAISS index ready
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 2 — RETRIEVE CONTEXT  (per new item)                      │
  │                                                                 │
  │  retriever.invoke(item["title"])                                │
  │    → top-3 most similar items from FAISS                       │
  │    → used as grounding context for the LLM                     │
  │                                                                 │
  │  Example context for "Allbirds Wool Runners":                  │
  │    • "Nike Air Max. Tags: shoes, running, athletic"            │
  │    • "Adidas Ultraboost. Tags: trail, running, outdoor"        │
  │    • "Aetna Bronze HSA. Tags: hsa, bronze, low-cost"           │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  context + item + user profile
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 3 — LLM CHAIN  (LCEL pipeline)                           │
  │                                                                 │
  │  prompt | llm | StrOutputParser()                              │
  │                                                                 │
  │  PromptTemplate:                                               │
  │    "Similar items: {context}                                   │
  │     New item: {title}, keywords: {keywords}                    │
  │     User: interests={interests}, segment={segment}             │
  │     Think step by step:                                        │
  │       1. What user intents does this satisfy?                  │
  │       2. What semantic tags beyond literal keywords?           │
  │       3. Why would this user engage?                           │
  │     Respond: INTENTS / TAGS / REASON"                         │
  │                                                                 │
  │  LLM options (swap model_name= to change):                     │
  │    gemma2:2b   → fastest, 2 GB RAM   (Google, distilled)       │
  │    gemma2      → best quality, 6 GB  (Google, distilled)       │
  │    llama3      → 5 GB RAM            (Meta)                    │
  │    mistral     → 5 GB RAM            (best for RAG tasks)      │
  │    phi3        → 3 GB RAM            (Microsoft, very fast)    │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │  INTENTS + TAGS + REASON
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 4 — APPLY ENRICHMENT                                      │
  │                                                                 │
  │  item["kw"]  +=  enriched_tags         (expand keyword index)  │
  │  item["intents"] = intents             (new feature for ranker)│
  │  ui_explanation  = reason              (shown to user)         │
  │                                                                 │
  │  Result: "footwear" now matches "shoes" queries                │
  │          "young adult coverage" matches "Aetna Bronze HSA"     │
  └─────────────────────────────────────────────────────────────────┘

  KV CACHE at inference time (TurboQuant — Google DeepMind, ICLR 2026):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Normal KV cache:   16-bit floats  →  large memory footprint    │
  │  TurboQuant:        16-bit → 3-bit via vector rotation          │
  │  Benefit:           6× memory reduction, 8× faster on H100     │
  │  Use case:          Long context (50+ candidates per request)   │
  │  Note:              Separate from GGUF weight compression       │
  │                     Both can be used together                   │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 8. Semantic Matching Upgrade Path

```
  STAGE 1 ──────────────────────────────────────────────────────────
  TF-IDF / Inverted Index  (this codebase)

  query: ["running", "shoes"]
  → exact term lookup in inverted index
  → TF-IDF score = Σ (count/length) × log(N/df)
  ✓ Zero infra, fully explainable, fast
  ✗ "footwear" won't match "shoes"

  STAGE 2 ──────────────────────────────────────────────────────────
  BM25  (Elasticsearch, Anserini, openrtb-semantic-matcher)

  Improvements over TF-IDF:
  + TF saturation: BM25(tf) = tf(k+1) / (tf + k×(1−b+b×dl/avgdl))
    the 10th "shoes" matters much less than the 1st
  + Document length normalisation: short docs not penalised
  ✓ Still explainable, minimal infra
  ✗ Still vocabulary-dependent

  STAGE 3 ──────────────────────────────────────────────────────────
  Dense Vector Search  (FAISS, Qdrant, Pinecone, pgvector)

  item_vec  = encode("Nike Air Max Running Shoes")  → [768 floats]
  query_vec = encode("best footwear for jogging")   → [768 floats]
  score     = cosine_sim(query_vec, item_vec)        → 0.87

  "footwear" ≈ "shoes" in embedding space → match found
  ✓ Semantic understanding, language-agnostic
  ✗ Needs GPU/inference infra, less explainable

  STAGE 4 ──────────────────────────────────────────────────────────
  Hybrid: BM25 + Dense via RRF  (production standard)

  rrf_score = 1/(k + rank_bm25) + 1/(k + rank_dense)  where k=60

  Used in: openrtb-semantic-matcher (BM25 + Qdrant + RRF)
  ✓ Best of both: exact match + semantic
  ✓ RRF is parameter-free and robust

  STAGE 5 ──────────────────────────────────────────────────────────
  LLM Re-ranker on top-50 candidates

  prompt = "User: {profile}. Rank by fit: {candidates[:50]}"
  → LLM reasons step-by-step → reordered list

  ✓ Highest quality, handles nuance
  ✗ ~$0.01–0.05 per request — run on top-50 only, not full corpus

  STAGE 6 ──────────────────────────────────────────────────────────
  RAG Enrichment  (this codebase, offline)

  Before indexing each item:
    context = similar_items_from_faiss(item)
    enriched = llm(f"Given {context}, generate intents/tags for {item}")
  → embed enriched description instead of raw title

  Used in: openrtb-semantic-matcher (DSPy + Ollama teacher pipeline)
  ✓ One-time offline cost, permanent retrieval improvement
  ✓ Enables stage 1–4 to find items they'd otherwise miss

  ─────────────────────────────────────────────────────────────────
  This codebase:  Stage 1  (core) + Stage 3/6 (optional, via FAISS)
  Production:     Stage 4  (BM25 + Qdrant + RRF) + Stage 6 (DSPy)
```

---

## 9. Domain Applicability Matrix

The same components apply differently across industries. Check marks show which stages each domain uses.

```
  Component            AdTech   Health Ins.   E-commerce   Content   Jobs
  ───────────────────────────────────────────────────────────────────────
  ItemIndex (TF-IDF)     ✓          ✓             ✓           ✓        ✓
  Dense retrieval        ✓          ✓             ✓           ✓        ✓
  LinearRanker           ✓          ✓             ✓           ✓        ✓
  NeuralRanker           ✓          ✓             ✓           ✓        ✓
  Second-price auction   ✓          ✗             partial     ✗        ✗
  Top-K ranking          ✗          ✓             ✓           ✓        ✓
  TokenBucket pacing     ✓          ✓ (quota)     ✓ (promo)   ✗        ✗
  FrequencyCap           ✓          ✓             ✓ (annoy)   ✓        ✓
  Throttler              ✓          ✗             ✓           ✗        ✗
  CoT explainer          ✗          ✓             ✓           partial  ✓
  RAG enrichment         ✓          ✓             ✓           ✓        ✓
  ShadowScorer           ✓          ✓             ✓           ✓        ✓
  ───────────────────────────────────────────────────────────────────────
  Label (what = click)   click      enroll        purchase    watch    apply
  Value signal           bid $      premium $     price $     LTV $    fee $
```

---

## 10. Data Flow Summary

```
  OFFLINE (hours / days)                  ONLINE (milliseconds)
  ─────────────────────                   ──────────────────────

  Raw items (ads, plans)                  User request arrives
       │                                         │
  RAG Enricher                             ItemIndex.search()
  (LangChain + Ollama)                          │
       │                                  Traffic shaping gates
  Enriched keywords                       (Throttle → Pace → Cap)
  + intents + tags                               │
       │                                  Score candidates
  ItemIndex rebuild                       (Linear or Neural)
  (TF-IDF + FAISS)                               │
       │                                  Auction or Ranking
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              (domain-specific)
                                                 │
  Feedback logs                            Serve result
  (Kafka / DB)                                   │
       │                                   User feedback
  Nightly batch:                          (click / enroll)
  retrain full model                             │
       │                                  Online SGD update
  Promote if better                       (this request only)
  (shadow → A/B → live)                         │
                                          Log to Kafka →
                                          nightly batch →
                                          full retrain
```

---

## 11. KPI and SLO framing (for production translation)

Use this scorecard to evaluate readiness and tradeoffs:

- **Latency SLO:** p95 decision latency under target budget (project reports p50/p95 in benchmark artifact).
- **Fill quality:** no-fill rate and reserve-hit behavior under pacing/capping pressure.
- **Business outcomes:** RPM for auction domains; enrollment rate and top-1 fit score for recommendation domains.
- **Model quality:** champion vs challenger log-loss delta over fixed evaluation windows.
- **Safety guardrails:** frequency-cap violation rate and budget overspend incidents (target: zero).

Recommended measurement source in this repo:
- `artifacts/benchmark.json` generated by `benchmark.py`.

---

## 12. Explicit architecture decisions

### ADR-001: Single shared scoring core across domains

- **Decision:** Keep retrieval + scoring model domain-agnostic.
- **Rationale:** Maximizes reuse and reduces model/platform fragmentation.
- **Consequence:** Domain logic moves to decision policy (auction vs ranking).

### ADR-002: Champion/challenger shadow scoring before promotion

- **Decision:** New model scores in shadow mode before any live traffic ownership.
- **Rationale:** Reduces blast radius; preserves business stability.
- **Consequence:** Slower rollout, but safer production adoption.

### ADR-003: Optional semantic stack, mandatory lightweight core

- **Decision:** Keep core path dependency-light; make RAG dependencies opt-in.
- **Rationale:** Faster onboarding and deterministic local execution.
- **Consequence:** Baseline semantics are sparse without optional enrichment.

---

*Source code: [mini_adtech.py](mini_adtech.py)*
*Product overview: [PRODUCT.md](PRODUCT.md)*
*Production system: [openrtb-semantic-matcher](https://github.com/zizimind/openrtb-semantic-matcher)*
