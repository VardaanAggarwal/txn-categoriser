"""Turns a raw bank statement description into a stable, comparable form.

Bank descriptions mix the useful part (the merchant or person) with noise that changes on every
transaction: reference numbers, IFSC codes, masked card numbers, dates and channel markers.
Stripping that noise is what lets the same merchant map to the same cache key every month.
"""

import re
from dataclasses import dataclass
from typing import Literal

Direction = Literal["debit", "credit"]

_VPA = re.compile(r"([A-Za-z0-9._-]+)@[A-Za-z][A-Za-z0-9.]*")
_SEPARATORS = re.compile(r"[\s/\\|:,;*#()\[\]{}\-_.+]+")
_IFSC = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
_MASKED = re.compile(r"^[0-9X]*X{3,}[0-9X]*$")

# Checked in order: the first marker found decides the channel.
_CHANNEL_MARKERS: list[tuple[str, frozenset[str]]] = [
    ("atm", frozenset({"ATM", "NWD"})),
    ("ach", frozenset({"ACH", "NACH", "ECS"})),
    ("billpay", frozenset({"BIL", "BBPS"})),
    ("upi", frozenset({"UPI"})),
    ("neft", frozenset({"NEFT"})),
    ("imps", frozenset({"IMPS"})),
    ("rtgs", frozenset({"RTGS"})),
    ("card", frozenset({"POS", "PCD"})),
]

_NOISE = frozenset(
    {
        # channel markers and payment boilerplate
        "UPI", "NEFT", "IMPS", "RTGS", "ACH", "NACH", "ECS", "ATM", "NWD", "WDL", "POS", "PCD",
        "BIL", "ONL", "BBPS", "DR", "CR", "P2A", "P2M", "TO", "FROM", "TRANSFER", "PAYMENT",
        "PAY", "PAID", "PHONE", "SENT", "USING", "VIA", "REF", "TXN", "INB", "MOB", "NA", "APP",
        "COLLECT", "REQUEST", "MERCHANT",
        # bank codes and names that describe the rail, not the payee
        "YESB", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB", "IDIB", "INDB", "YES", "AXIS",
        "SBI", "STATE", "INDIA", "KOTAK", "MAHINDRA", "BANK",
        # months, so a monthly salary or rent keeps the same key
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV",
        "DEC",
    }
)


@dataclass(frozen=True)
class Normalised:
    raw: str
    channel: str
    direction: Direction | None
    tokens: tuple[str, ...]

    @property
    def key(self) -> str:
        """The payee and remark words with all per-transaction noise removed."""
        return " ".join(self.tokens)

    @property
    def cache_key(self) -> str:
        return f"{self.direction or 'unknown'}|{self.channel}|{self.key}"

    @property
    def features(self) -> str:
        """Text the ML model sees: channel and direction markers, then the cleaned key."""
        return f"__{self.channel}__ __{self.direction or 'unknown'}__ {self.key}"


def _clean_token(token: str) -> str:
    if _IFSC.match(token) or _MASKED.match(token):
        return ""
    digits = sum(ch.isdigit() for ch in token)
    if digits >= 4:  # reference numbers, account numbers, mandate IDs, years
        return ""
    token = "".join(ch for ch in token if not ch.isdigit())
    if len(token) < 2 or token in _NOISE:
        return ""
    return token


def _detect_channel(raw_tokens: list[str]) -> str:
    present = set(raw_tokens)
    for channel, markers in _CHANNEL_MARKERS:
        if present & markers:
            return channel
    return "other"


def _detect_direction(raw_tokens: list[str]) -> Direction | None:
    present = set(raw_tokens)
    if "DR" in present:
        return "debit"
    if "CR" in present:
        return "credit"
    return None


def normalise(description: str, direction: Direction | None = None) -> Normalised:
    # Keep only the local part of a UPI ID: "swiggy.food@ybl" -> "swiggy.food".
    text = _VPA.sub(lambda m: f" {m.group(1)} ", description).upper()
    raw_tokens = [t for t in _SEPARATORS.split(text) if t]
    tokens = tuple(t for t in (_clean_token(t) for t in raw_tokens) if t)
    return Normalised(
        raw=description,
        channel=_detect_channel(raw_tokens),
        direction=direction or _detect_direction(raw_tokens),
        tokens=tokens,
    )
