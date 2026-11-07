# txn-categoriser

Categorises Indian bank and UPI transactions, such as
`UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/Payment from Phone`, into spending categories. It
uses a cascade: cheap, predictable layers go first, an LLM handles only what they can't, and
anything nobody is confident about goes to a review queue instead of being guessed.

```
description ─► normalise ─► cache ─► rules ─► ML model ─► LLM (optional) ─► review queue
                             hit?    match?   confident?  valid and
                                                          confident?
                                                              ▲
                                       retrieval (optional) ──┘
                          vector index: merchant facts + labelled examples + corrections
```

Every result says which layer decided it and why, for example `merchant:SWIGGY`, `ml:p=0.95` or
`llm:ollama`. When a user corrects a category, the correction is stored and that merchant gets the
corrected category from then on, ahead of every other layer.

## Why a cascade

| Layer | Cost | Strength | Weakness |
|---|---|---|---|
| Rules | ~free | Exact and explainable | Only knows merchants someone wrote down |
| ML (TF-IDF + logistic regression) | ~30K transactions/sec on a laptop | Generalises from words like DHABA, KIRANA, MEDICOS | Can't know an unseen brand like THEOBROMA |
| LLM | Seconds per batch | Knows brands it has never seen in this data | Slow, can return invalid output |

In a finance app a confident wrong answer is worse than "not sure", so each layer only answers
above a confidence threshold, and the rest falls through.

## Results

From `python -m txncat eval`, on 1,320 synthetic transactions (see
[Limitations](#limitations)). Full output: [results/RESULTS.md](results/RESULTS.md).

**Unseen merchants** — no merchant or person in the test set appears in training:

| Mode | Auto-resolved | Precision on auto-resolved |
|---|---|---|
| Rules only | 43.6% | 100.0% |
| ML only | 100.0% | 46.1% |
| **Rules + ML (threshold 0.6)** | **57.4%** | **97.0%** |

**Random split** — merchants mostly seen before, like a returning user's statements:

| Mode | Auto-resolved | Precision on auto-resolved |
|---|---|---|
| Rules + ML (threshold 0.6) | 96.0% | 100.0% |

- *Auto-resolved* is the share categorised without going to review; *precision* is the share of
  those that were right.
- Raising the ML threshold trades coverage for precision: 80% auto-resolved at 80% precision at
  0.3, 57% at 97% at 0.6, 50% at 100% at 0.8.
- The remaining 43% of unseen-merchant transactions is the LLM layer's job. **The LLM layer has
  not yet been measured against a real model**; see below for how to run it.
- Throughput: about 30,000 transactions/second for rules + ML on an Apple M1 laptop.
- Cache: 30% of transactions that got past the rules were served from cache when processed one at
  a time, because normalisation gives a merchant the same key every time.

## How the LLM layer is kept honest

- **Strict output.** The reply must be JSON with an `id`, a `category` from the fixed list and a
  `confidence` between 0 and 1, validated with Pydantic. Unknown categories, out-of-range
  confidence and IDs that weren't asked about are rejected.
- **Targeted re-ask.** Only the items that came back invalid are asked again, once. If they fail
  again, they go to review.
- **Retries.** Timeouts, connection errors, HTTP 429 and 5xx are retried with exponential backoff
  and jitter. Other errors fail fast.
- **Retrieved few-shot examples.** The prompt includes the most similar labelled transactions,
  found with the same TF-IDF vectors the ML model uses, or with hybrid search when RAG is on
  (below).
- **Batching and deduplication.** Up to 20 transactions per call, and repeated descriptions are
  sent once. Descriptions with no usable words, such as transfers to phone-number UPI IDs, are
  never merged, because they'd all share the same empty key.
- **Prompt injection.** Descriptions are quoted as JSON strings and the system prompt says to treat
  them as data. Because output is limited to the category list, an injected description can at
  worst mislabel itself.
- **Providers.** One interface, two clients: [Ollama](https://ollama.com) for free local models
  and Gemini, which has a free tier. Tests use a scripted fake, so they need no network.

## Retrieval-augmented prompts (RAG)

The LLM knows big brands but not every regional chain, and a 3B local model knows far fewer. With
RAG on, each batch sent to the LLM carries retrieved context:

- **Merchant facts** from a hand-written knowledge base
  ([data/merchant_kb.jsonl](data/merchant_kb.jsonl), 88 Indian merchants), for example
  `"THEOBROMA": Bakery and dessert cafe chain -> food_dining`.
- **Labelled examples**: training transactions and every user correction.

How it works:

| Piece | What it does |
|---|---|
| Embeddings ([embeddings.py](txncat/embeddings.py)) | One interface, three backends: a hashing embedder (offline, no model, lexical), Ollama (`nomic-embed-text`, local, semantic) and Gemini (`gemini-embedding-001`, free tier, semantic). Queries and documents are embedded with the task type or prefix each model expects. |
| Vector store ([vectorstore.py](txncat/vectorstore.py)) | Documents and float32 vectors in SQLite, searched exactly with one numpy matrix multiply. It records which embedder built it and refuses vectors from any other, since cross-model similarities are meaningless. |
| Hybrid search ([rag.py](txncat/rag.py)) | Examples come from dense (embedding) search and the ML layer's TF-IDF search, merged with reciprocal rank fusion, so neither similarity scale has to be calibrated against the other. |
| Similarity floor | Facts and examples below a minimum similarity are dropped. Irrelevant context is worse than none, because small models copy the label of a wrong "similar" example. |
| Corrections | A correction goes into the cache (exact merchant, highest priority) **and** the vector index, so it's also used as an example for lookalike merchants. |
| Failure handling | If the embedding API fails, the batch is sent without context instead of failing. Embedding calls are retried like LLM calls, and a batch's queries are embedded once even though examples and facts are both retrieved. |

```bash
python -m txncat index --embedder hashing          # builds models/index.db; or --embedder ollama
python -m txncat categorise "UPI/DR/412345678901/THEOBROMA/YESB/theobroma@ybl/NA" \
  --direction debit --llm ollama --rag hashing
python -m txncat eval --llm ollama --rag ollama    # adds a rules+ml+llm+rag row to RESULTS.md
```

In the API, set `TXNCAT_RAG=hashing|ollama|gemini` (with an LLM enabled), and optionally
`TXNCAT_INDEX=models/index.db` so the index survives restarts.

**What's not known yet:** RAG has not been run against a real model, so whether it improves
accuracy, and by how much, is unmeasured. The similarity floors (0.35 for examples, 0.5 for facts)
are starting values, not tuned, and the right values differ per embedder. In the evaluation, the
index contains training rows only, so no test labels leak, but the knowledge base was written by
the same person as the synthetic data and covers many of its merchants. That mode measures the LLM
with a good merchant directory, not unaided generalisation.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                  # no network needed
python -m txncat generate               # writes data/transactions.csv
python -m txncat eval                   # writes results/RESULTS.md and metrics.json
python -m txncat train                  # saves models/ml.joblib
python -m txncat categorise "POS 4XXXXXXXXXXX1234 CROMA" --direction debit
```

API:

```bash
uvicorn txncat.api:app --reload
curl -s localhost:8000/v1/categorise -H 'content-type: application/json' \
  -d '{"transactions": [{"description": "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA", "direction": "debit"}]}'

# Correct a category; the same merchant is categorised this way from now on
curl -s localhost:8000/v1/corrections -H 'content-type: application/json' \
  -d '{"transaction": {"description": "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA", "direction": "debit"}, "category": "groceries"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/categorise` | Categorise 1–500 transactions |
| `POST /v1/corrections` | Store a user's correction |
| `GET /healthz` | Liveness and enabled layers |

Docker:

```bash
docker build -t txn-categoriser .
docker run -p 8000:8000 txn-categoriser
```

### Measuring the LLM layer for free

```bash
brew install ollama && ollama serve &   # or see ollama.com for Linux
ollama pull qwen2.5:3b                  # about 2 GB, runs on an 8 GB laptop
python -m txncat eval --llm ollama
```

Or set `GEMINI_API_KEY` from Google AI Studio and use `--llm gemini`. To use the LLM in the API,
set `TXNCAT_LLM=ollama` or `TXNCAT_LLM=gemini`.

## Layout

| Path | What it does |
|---|---|
| `txncat/normalise.py` | Strips reference numbers, IFSC codes, masked cards, months; detects channel and direction |
| `txncat/rules.py` | Channel, keyword and known-merchant rules |
| `txncat/ml.py` | TF-IDF character n-grams + logistic regression; nearest-example retrieval |
| `txncat/llm_categoriser.py` | Prompting, batching, validation and the re-ask |
| `txncat/llm/` | Provider interface, Ollama and Gemini clients, test fake |
| `txncat/embeddings.py` | Hashing, Ollama and Gemini embedders |
| `txncat/vectorstore.py` | SQLite-backed vector store with exact cosine search |
| `txncat/knowledge.py` | Loads and validates the merchant knowledge base |
| `txncat/rag.py` | Hybrid retrieval (dense + TF-IDF, reciprocal rank fusion), indexing, corrections |
| `txncat/retry.py` | Exponential backoff with jitter |
| `txncat/cache.py` | SQLite cache keyed on the normalised description |
| `txncat/pipeline.py` | The cascade |
| `txncat/dataset.py` | Synthetic data generator and train/test splits |
| `txncat/evaluate.py` | Metrics and the results report |
| `txncat/api.py` | FastAPI service |

## Limitations

- **The data is synthetic.** It's generated from templates modelled on real statement formats,
  with made-up names and numbers. Real statements are messier, so real accuracy will be lower.
- **The rules and the data were written by the same person**, which flatters the rules layer. Its
  100% precision would not survive real data.
- **The LLM layer is untested against a real model.** Its logic is covered by tests with a fake
  client, but its accuracy, latency and cost are unknown until the commands above are run.
- **One labelled category per transaction.** Split payments and refunds are not modelled.
- **Corrections are global and don't retrain the model.** They're stored per merchant for the
  whole deployment, not per user. With RAG on they're used as LLM examples, but they aren't fed
  back into the ML model's training data yet.
- **Changing the embedder means a new index.** Vectors from two models can't be compared, so the
  store refuses to mix them. Corrections still live in the cache, but they aren't re-embedded
  automatically.
- **No authentication or rate limiting** on the API.

## Licence

MIT
