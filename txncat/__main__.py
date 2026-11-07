"""Command line: python -m txncat {generate,train,index,eval,categorise}"""

import argparse
import json
import sys
from pathlib import Path

from .dataset import generate, load_csv, write_csv
from .embeddings import EMBEDDERS, make_embedder
from .evaluate import run_all, train_ml, write_report
from .knowledge import DEFAULT_KB
from .llm import PROVIDERS, make_client
from .llm_categoriser import LLMCategoriser
from .ml import MLCategoriser
from .pipeline import Pipeline, Transaction
from .rag import build_retriever

DEFAULT_DATA = "data/transactions.csv"
DEFAULT_MODEL = "models/ml.joblib"
DEFAULT_INDEX = "models/index.db"


def _add_rag_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rag", choices=EMBEDDERS,
                   help="retrieve examples and merchant facts with this embedder")
    p.add_argument("--embed-model", help="model name for the embedder")
    p.add_argument("--kb", default=DEFAULT_KB, help="merchant knowledge base (JSONL)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="txncat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write the synthetic labelled dataset")
    p.add_argument("--out", default=DEFAULT_DATA)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="train the ML model on the full dataset")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=DEFAULT_MODEL)

    p = sub.add_parser("index", help="embed the dataset and knowledge base into a vector index")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--kb", default=DEFAULT_KB)
    p.add_argument("--out", default=DEFAULT_INDEX)
    p.add_argument("--embedder", choices=EMBEDDERS, default="hashing")
    p.add_argument("--embed-model")
    p.add_argument("--refresh", action="store_true",
                   help="re-embed training data and facts (user corrections are kept)")

    p = sub.add_parser("eval", help="evaluate every layer and write results/")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default="results")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--llm", choices=PROVIDERS, help="also evaluate the LLM layer")
    p.add_argument("--llm-model", help="model name for the LLM provider")
    _add_rag_args(p)

    p = sub.add_parser("categorise", help="categorise one description")
    p.add_argument("description")
    p.add_argument("--direction", choices=["debit", "credit"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llm", choices=PROVIDERS)
    p.add_argument("--llm-model")
    _add_rag_args(p)
    p.add_argument("--index", default=DEFAULT_INDEX, help="vector index built by `index`")

    args = parser.parse_args(argv)

    if getattr(args, "rag", None) and not args.llm:
        print("--rag needs --llm: retrieval only feeds the LLM layer", file=sys.stderr)
        return 2

    if args.command == "generate":
        rows = generate(args.seed)
        write_csv(rows, args.out)
        print(f"Wrote {len(rows)} rows to {args.out}")
    elif args.command == "train":
        train_ml(load_csv(args.data)).save(args.out)
        print(f"Saved model to {args.out}")
    elif args.command == "index":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        retriever = build_retriever(
            make_embedder(args.embedder, args.embed_model), index_path=args.out,
            rows=load_csv(args.data), kb_path=args.kb, refresh=args.refresh,
        )
        store = retriever.store
        print(f"{args.out}: {store.count(source='train')} training examples, "
              f"{store.count(source='kb')} merchant facts, {store.count(source='user')} corrections "
              f"({store.embedder_name})")
    elif args.command == "eval":
        client = make_client(args.llm, args.llm_model) if args.llm else None
        embedder = make_embedder(args.rag, args.embed_model) if args.rag else None
        report = run_all(load_csv(args.data), ml_threshold=args.threshold, llm_client=client,
                         embedder=embedder, kb_path=args.kb)
        write_report(report, args.out)
        print(f"Wrote {args.out}/RESULTS.md and {args.out}/metrics.json")
    elif args.command == "categorise":
        try:
            ml = MLCategoriser.load(args.model)
        except FileNotFoundError:
            print(f"No model at {args.model}. Run: python -m txncat train", file=sys.stderr)
            return 1
        llm = retriever = None
        if args.llm and args.rag:
            # Reuses the persisted index; builds it first if it's empty.
            Path(args.index).parent.mkdir(parents=True, exist_ok=True)
            retriever = build_retriever(
                make_embedder(args.rag, args.embed_model), index_path=args.index,
                lexical_fn=ml.nearest,
                rows=load_csv(DEFAULT_DATA) if Path(DEFAULT_DATA).exists() else None,
                kb_path=args.kb,
            )
            llm = LLMCategoriser(make_client(args.llm, args.llm_model),
                                 examples_fn=retriever.examples, facts_fn=retriever.facts)
        elif args.llm:
            llm = LLMCategoriser(make_client(args.llm, args.llm_model), examples_fn=ml.nearest)
        result = Pipeline(ml=ml, llm=llm, retriever=retriever).categorise(
            [Transaction(description=args.description, direction=args.direction)]
        )[0]
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
