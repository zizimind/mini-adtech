# MiniAdTech — Product Overview

**What this is, what the outputs mean, and why it matters.**

---

## The One-Line Pitch

Every time a user sees an ad, gets a health plan recommendation, or receives a product suggestion, the same invisible pipeline runs in a tight decision budget. This project strips that pipeline down to its bare essentials — in a single Python module (~950 lines) — so anyone can understand it, adapt it, and own it.

---

## The Problem It Solves

Large companies (Google, Amazon, insurance marketplaces, Netflix) spend hundreds of millions of dollars building systems that answer one question:

> **Given this user, right now — what is the best item to show them?**

That question is harder than it looks:

- There are millions of items to choose from (ads, products, health plans, movies)
- Users are all different (age, interests, budget, behaviour)
- Suppliers compete for attention (advertisers bid, insurers want enrollments)
- Showing the wrong item too many times destroys trust
- Budgets must be spent evenly — not all in the first hour

This project implements all of that, end to end, for two real industries.

### Scope and evidence standards

- This repo is an educational reference implementation, not a production ad-serving stack.
- All datasets are synthetic and in-repo.
- Performance and quality claims should be taken from generated artifacts (`artifacts/benchmark.json`) and test output, not narrative text alone.

---

## The Two Industries Covered

### Industry 1 — Digital Advertising (AdTech)

**Business context:** Every time a webpage loads, a silent auction runs. Advertisers (Nike, Apple, Netflix) bid in real time for the right to show you their ad. The auction happens in ~100ms. The winner pays the second-highest bid, not their own.

**Why second-highest bid?** It is the only pricing rule where bidding your true value is the dominant strategy. Google, Amazon, and all major ad platforms use this. It is called the Vickrey auction (Nobel Prize, 1996).

**What the system optimises:**
- Revenue per impression (eCPM = predicted click rate × bid × 1000)
- Budget smoothness (no advertiser burns their daily budget in the first hour)
- User experience (no advertiser shown more than 2× to the same user)

---

### Industry 2 — Health Insurance Recommendation

**Business context:** A person visits an insurance marketplace (Healthcare.gov, Oscar Health, Stride). They describe their needs. The platform must recommend the best 3 plans from hundreds of options.

**No auction here.** There is no bidding — just ranking by fit. The same retrieval and scoring engine runs, but instead of a second-price auction, it returns a ranked list with a plain-English explanation.

**What the system optimises:**
- Fit score (how well a plan matches the patient's stated needs)
- Explanation quality (why this plan — chain-of-thought reasoning)
- Enrollment prediction (which plans the patient is most likely to choose)

---

## The Three Outputs — What They Mean

### Simulation A Output — AdTech Auction

```
#    user  query               winner      adv          $price  click
  1  u4    music+streaming     a06         netflix     $ 0.890  ·
  7  u3    laptop+pro          a03         apple       $ 1.855  ·
```

| Column | What it means | Why it matters |
|--------|--------------|----------------|
| `user` | Which user profile triggered the request | Different users see different ads |
| `query` | What the page/search was about | Retrieval matches ads to context |
| `winner` | Which ad won the auction | The highest predicted-value ad |
| `adv` | Which advertiser won | Revenue attribution |
| `$price` | What the advertiser pays (NOT their bid) | Second-price: always less than their bid |
| `click ✓/·` | Did the user click? (simulated) | The feedback signal for learning |

```
Revenue: $16.24  |  No-fill: 39/60
```

- **Revenue** — total money collected from all auctions in this session
- **No-fill** — requests where no ad was shown (pacing, frequency cap, or reserve price not met). 39/60 is high here because the simulation runs a compressed 1-hour window with tight caps. In production, no-fill below 5% is the target.

```
Advertiser    Wins   Spent    $/win
nike             7   $5.40   $0.771
apple            2   $3.51   $1.755
```

- **Wins** — how many auctions this advertiser won
- **Spent** — total cleared price (always < their bid × wins)
- **$/win** — effective cost per impression. Apple pays $1.75/win because they face stronger competition in laptop queries.

```
Shadow Scoring Report  (n=21)
Champion  (linear) avg log-loss : 0.5407
Challenger (neural) avg log-loss: 0.7146
Decision: champion → keep
```

This is the **A/B test result**. Two models scored every auction:
- **Champion (logistic regression)** — simpler, currently live
- **Challenger (neural network)** — more complex, being evaluated

Log-loss measures how well each model's probability estimates matched actual clicks. Lower = better. Here the simpler model wins on 21 samples — in production you'd run this on millions before promoting the challenger.

---

### Simulation B Output — Health Insurance Recommendation

```
Patient p5 (budget_conscious) | query: affordable+young
  1. [h03] Aetna Bronze HSA    score=35.8  Fit 36/100 — matches hsa, low-cost; partial fit
  2. [h06] Humana Bronze HMO   score=33.2  Fit 33/100 — matches low-cost; partial fit
  3. [h02] Kaiser Gold HMO     score=32.5  Fit 33/100 — strong overall fit
```

| Element | What it means |
|---------|--------------|
| `Patient p5 (budget_conscious)` | User profile — drives personalisation |
| `query: affordable+young` | What they searched for |
| `score=35.8` | Fit score out of 100: semantic match × quality × personalisation |
| `Fit 36/100` | Human-readable version of the score |
| The explanation | Chain-of-thought reasoning: *why* this plan fits this patient |

**Why the scores are moderate (30–45/100):** The model starts with random weights and only sees 15 patients before printing results. In production, after thousands of enrollments and feedback signals, scores would be higher and more discriminating. The last two lines show the model learning:

```
Ranker weights after 15 patients:
w = ['-0.061', '-0.121', '-0.101', '+0.099']
features: [semantic_sim, value_norm, quality, user_affinity]
```

`user_affinity` (+0.099) is the only positive weight — the model is learning that matching a user's explicit interests matters most. All other features are weakly negative because low quality/value plans were shown when no better match was available.

---

### Simulation C Output — RAG Enrichment

```
Item    : Humana Bronze HMO
User    : budget_conscious | interests: hsa, low-cost, high-deductible
Intents : find low-cost
Tags    : hmo, low-cost, basic, recommended, humana, bronze, young
Reason  : keyword overlap on [low-cost]
```

This is the **pre-processing step** that makes retrieval smarter. Before an item is indexed, an AI reads it and generates:

| Field | What it adds | Business value |
|-------|-------------|----------------|
| `Intents` | Why a user would want this | Matches intent, not just keywords |
| `Tags` | Semantic synonyms + categories | Finds "footwear" when user typed "shoes" |
| `Reason` | Plain-English match explanation | Used in UI to explain recommendations |

**When Ollama is running** (a local AI model), these fields are generated by Gemma 2, Llama 3, or Mistral using chain-of-thought reasoning. When it is not running, a rule-based fallback handles it — same interface, less intelligence.

```
LLM Catalog (ollama pull <model>)
  gemma2:2b   Google   2B    2 GB   fits on any laptop, fast
  gemma2      Google   9B    6 GB   best quality/size; default
  llama3      Meta     8B    5 GB   strong general reasoning
  mistral     Mistral  7B    5 GB   excellent for RAG + retrieval
```

These are open-source AI models that run **locally on your machine** — no API key, no cloud cost, no data leaving your network. Critical for health data and any PII.

---

## The Core Insight — One Engine, Every Domain

The most important thing to understand:

> The retrieval, scoring, and learning pipeline is **identical** across every domain.
> Only the **decision layer** changes.

```
AdTech:            retrieve → score → AUCTION  → one winner pays
Health Insurance:  retrieve → score → RANK TOP-3 → explain why
E-commerce:        retrieve → score → RANK + PRICE SORT
Content (Netflix): retrieve → score → RANK + DIVERSITY FILTER
Job Matching:      retrieve → score → RANK + MUTUAL FIT CHECK
```

The last line of Simulation B says it explicitly:

> *"The ranker does not know which domain it is in."*

This is the business opportunity: one ML platform, many verticals.

---

## The AI Stack — Explained Simply

| Layer | Technology | Plain-English role |
|-------|-----------|-------------------|
| Keyword search | TF-IDF (built-in) | Find items that mention the right words |
| Semantic search | FAISS + sentence-transformers | Find items that *mean* the right thing |
| Click/enroll prediction | Logistic regression → MLP | Estimate how likely a user is to engage |
| Enrichment | LangChain + Gemma2/Llama3 | AI reads items and adds smarter tags |
| Budget control | Token bucket algorithm | Spend evenly across the day |
| User protection | Frequency cap | Never annoy users with repeat ads |
| Model testing | Shadow scoring | Safely test new AI models in production |

---

## The Production Version

This educational engine demonstrates the same concepts as the production system:

**[openrtb-semantic-matcher](https://github.com/zizimind/openrtb-semantic-matcher)**

| | This project | Production |
|---|---|---|
| Language | Python (educational) | Go (250+ QPS) |
| Search | TF-IDF | BM25 + Qdrant vector DB |
| LLM enrichment | LangChain + Ollama | DSPy + Ollama (offline) |
| Caching | None | Redis (<1ms hits) |
| Budget updates | Simulated | NATS JetStream (real-time) |
| Logging | Print | Kafka + ZSTD compression |

---

## Staff-level review checklist (included)

For technical reviewers, the project now includes explicit coverage of:

- **Correctness:** unit tests with edge-case scenarios.
- **Reproducibility:** Docker, Docker Compose, and deterministic benchmark runner.
- **Measurability:** artifact metrics for no-fill, RPM, enrollment rate, fit score, and p50/p95 latency.
- **Operational thinking:** pacing, frequency cap, throttling, and shadow model workflow.
- **Risk disclosure:** known limitations documented in `README.md`.

---

## Who Should Read This

| Audience | What to focus on |
|----------|-----------------|
| Product Managers | This document + the two simulation outputs |
| Business Analysts | The revenue, no-fill, and $/win tables |
| Data Scientists | Sections 3–4 (LinearRanker, NeuralRanker) + ShadowScorer |
| ML Engineers | Section 9 (semantic ladder) + RAGEnricher |
| Engineers | Full source + README for setup |

---

*System design & architecture diagrams: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)*
*Article companion: [Medium — The Universal Matching Machine](#) (coming soon)*
*Production code: [github.com/zizimind/openrtb-semantic-matcher](https://github.com/zizimind/openrtb-semantic-matcher)*
