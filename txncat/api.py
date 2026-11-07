"""HTTP API. Run with: uvicorn txncat.api:app

Configuration (environment variables):
- TXNCAT_MODEL: path to a trained model (default models/ml.joblib). If missing, a model is
  trained from TXNCAT_DATA at startup.
- TXNCAT_DATA: labelled CSV (default data/transactions.csv).
- TXNCAT_LLM: "ollama", "gemini" or "none" (default none).
- TXNCAT_LLM_MODEL: model name for the chosen provider.
- TXNCAT_CACHE: SQLite path for the cache (default: in memory).
- TXNCAT_RAG: embedder for retrieval-augmented prompts: "hashing", "ollama", "gemini" or "none"
  (default none). Only used when an LLM is enabled.
- TXNCAT_EMBED_MODEL: model name for the chosen embedder.
- TXNCAT_INDEX: SQLite path for the vector index (default: in memory, rebuilt at startup).
- TXNCAT_KB: merchant knowledge base (default data/merchant_kb.jsonl).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .cache import Cache
from .categories import Category
from .dataset import load_csv
from .embeddings import make_embedder
from .evaluate import train_ml
from .knowledge import DEFAULT_KB
from .llm import make_client
from .llm_categoriser import LLMCategoriser
from .ml import MLCategoriser
from .pipeline import CategorisedTransaction, Pipeline, Transaction
from .rag import build_retriever

MAX_BATCH = 500


class CategoriseRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class CategoriseResponse(BaseModel):
    results: list[CategorisedTransaction]


class CorrectionRequest(BaseModel):
    transaction: Transaction
    category: Category


def pipeline_from_env() -> Pipeline:
    model_path = Path(os.environ.get("TXNCAT_MODEL", "models/ml.joblib"))
    data_path = Path(os.environ.get("TXNCAT_DATA", "data/transactions.csv"))
    rows = load_csv(data_path) if data_path.exists() else None
    if model_path.exists():
        ml = MLCategoriser.load(model_path)
    elif rows:
        ml = train_ml(rows)
    else:
        ml = None

    llm = retriever = None
    provider = os.environ.get("TXNCAT_LLM", "none")
    if provider != "none":
        client = make_client(provider, os.environ.get("TXNCAT_LLM_MODEL"))
        embedder_name = os.environ.get("TXNCAT_RAG", "none")
        if embedder_name != "none":
            kb_path = Path(os.environ.get("TXNCAT_KB", DEFAULT_KB))
            retriever = build_retriever(
                make_embedder(embedder_name, os.environ.get("TXNCAT_EMBED_MODEL")),
                index_path=os.environ.get("TXNCAT_INDEX", ":memory:"),
                lexical_fn=ml.nearest if ml else None,
                rows=rows,
                kb_path=kb_path if kb_path.exists() else None,
            )
            llm = LLMCategoriser(client, examples_fn=retriever.examples, facts_fn=retriever.facts)
        else:
            llm = LLMCategoriser(client, examples_fn=ml.nearest if ml else None)

    return Pipeline(
        ml=ml, llm=llm, retriever=retriever,
        cache=Cache(os.environ.get("TXNCAT_CACHE", ":memory:")),
    )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or pipeline_from_env()
        yield

    app = FastAPI(title="txn-categoriser", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        return {"status": "ok", "layers": request.app.state.pipeline.layers}

    @app.post("/v1/categorise", response_model=CategoriseResponse)
    def categorise(body: CategoriseRequest, request: Request) -> CategoriseResponse:
        # A sync endpoint: FastAPI runs it in a worker thread, so a slow LLM call doesn't block
        # the event loop.
        return CategoriseResponse(results=request.app.state.pipeline.categorise(body.transactions))

    @app.post("/v1/corrections")
    def correct(body: CorrectionRequest, request: Request) -> dict:
        if not request.app.state.pipeline.record_correction(body.transaction, body.category):
            raise HTTPException(
                status_code=422,
                detail="cannot store this correction: no cache, or no usable words in the description",
            )
        return {"stored": True}

    return app


app = create_app()
