"""The merchant knowledge base: short, hand-written facts about Indian merchants.

The LLM knows big brands but not every regional chain or app, and small models know even fewer.
Retrieving a one-line fact ("THEOBROMA: bakery and dessert chain") for the merchant in front of it
is cheaper and more reliable than hoping the model remembers.

Each line of the JSONL file is one merchant:
    {"name": "THEOBROMA", "category": "food_dining", "note": "Bakery and dessert chain"}
with optional "aliases" for other spellings that appear in statements.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .categories import Category

DEFAULT_KB = "data/merchant_kb.jsonl"  # relative to the working directory, like the dataset


class _FactModel(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    category: Category
    note: str = Field(min_length=3, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)


@dataclass(frozen=True)
class MerchantFact:
    name: str
    category: Category
    note: str
    aliases: tuple[str, ...] = ()

    @property
    def search_text(self) -> str:
        """What gets embedded: the names a statement might show, not the note.

        Queries are bank descriptions, which contain merchant names, not prose. Embedding the note
        too would pull "Bakery and dessert chain" towards every description with "CHAIN" in it.
        """
        return " ".join((self.name, *self.aliases))


@dataclass(frozen=True)
class MerchantNote:
    """A retrieved fact, as it is shown to the LLM."""

    name: str
    category: Category
    note: str
    score: float


def load_kb(path: str | Path = DEFAULT_KB) -> list[MerchantFact]:
    """Load and validate the knowledge base. A bad line is an error, not a silent skip."""
    facts: list[MerchantFact] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                model = _FactModel.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{number}: invalid knowledge-base entry: {exc}") from exc
            name = model.name.upper()
            if name in seen:
                raise ValueError(f"{path}:{number}: duplicate merchant {name!r}")
            seen.add(name)
            facts.append(MerchantFact(
                name=name,
                category=model.category,
                note=model.note,
                aliases=tuple(a.upper() for a in model.aliases),
            ))
    return facts
