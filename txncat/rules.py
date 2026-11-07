"""Deterministic rules: the cheapest, most predictable layer.

Rules run before any model. Order matters:
1. Channel rules (an ATM withdrawal is always cash).
2. Keyword rules (SALARY, EMI, RENT...), which can depend on direction.
3. Known merchants, matched on whole tokens or token pairs, debits only, because a credit from a
   merchant is usually a refund or cashback, not spending.
"""

import re
from dataclasses import dataclass

from .categories import Category as C
from .normalise import Normalised


@dataclass(frozen=True)
class RuleMatch:
    category: C
    reason: str


_KEYWORDS: list[tuple[re.Pattern[str], C, str | None]] = [
    (re.compile(r"\b(SALARY|SAL)\b"), C.INCOME, "credit"),
    (re.compile(r"\b(INT PD|INTEREST|DIVIDEND|DIV)\b"), C.INCOME, "credit"),
    (re.compile(r"\b(EMI|LOAN)\b"), C.EMI_LOANS, "debit"),
    (re.compile(r"\b(SIP|MUTUAL FUND|MF)\b"), C.INVESTMENTS, "debit"),
    (re.compile(r"\b(RENT|RENTPAY)\b"), C.RENT, "debit"),
    (
        re.compile(r"\b(RECHARGE|ELECTRICITY|BROADBAND|POSTPAID|WATER BILL|GAS BILL)\b"),
        C.BILLS_UTILITIES,
        "debit",
    ),
    (
        re.compile(r"\b(PHARMACY|MEDICAL|MEDICOS|CHEMIST|HOSPITAL|CLINIC|DIAGNOSTICS|PATHLABS?)\b"),
        C.HEALTH,
        "debit",
    ),
    (re.compile(r"\b(KIRANA|PROVISION|GENERAL STORE|SUPERMARKET)\b"), C.GROCERIES, "debit"),
    (re.compile(r"\b(PETROL|FUEL)\b"), C.TRANSPORT, "debit"),
    (re.compile(r"\b(RESTAURANT|DHABA|CAFE)\b"), C.FOOD_DINING, "debit"),
]

# A deliberately partial list of well-known merchants. Anything not here falls through to the
# ML model, which is the point: rules are precise but never complete.
_MERCHANTS: dict[str, C] = {
    "SWIGGY": C.FOOD_DINING, "ZOMATO": C.FOOD_DINING, "DOMINOS": C.FOOD_DINING,
    "MCDONALDS": C.FOOD_DINING, "KFC": C.FOOD_DINING, "STARBUCKS": C.FOOD_DINING,
    "HALDIRAMS": C.FOOD_DINING,
    "SWIGGY INSTAMART": C.GROCERIES, "INSTAMART": C.GROCERIES, "BLINKIT": C.GROCERIES,
    "ZEPTO": C.GROCERIES, "BIGBASKET": C.GROCERIES, "DMART": C.GROCERIES, "JIOMART": C.GROCERIES,
    "UBER": C.TRANSPORT, "OLA": C.TRANSPORT, "RAPIDO": C.TRANSPORT, "IRCTC": C.TRANSPORT,
    "DMRC": C.TRANSPORT, "INDIGO": C.TRANSPORT, "FASTAG": C.TRANSPORT,
    "AMAZON": C.SHOPPING, "FLIPKART": C.SHOPPING, "MYNTRA": C.SHOPPING, "AJIO": C.SHOPPING,
    "NYKAA": C.SHOPPING, "MEESHO": C.SHOPPING,
    "AIRTEL": C.BILLS_UTILITIES, "JIO": C.BILLS_UTILITIES, "BSES": C.BILLS_UTILITIES,
    "TATA POWER": C.BILLS_UTILITIES, "ACT FIBERNET": C.BILLS_UTILITIES,
    "BAJAJ FINANCE": C.EMI_LOANS, "TATA CAPITAL": C.EMI_LOANS, "HOME CREDIT": C.EMI_LOANS,
    "ZERODHA": C.INVESTMENTS, "GROWW": C.INVESTMENTS, "UPSTOX": C.INVESTMENTS,
    "KUVERA": C.INVESTMENTS,
    "NETFLIX": C.ENTERTAINMENT, "SPOTIFY": C.ENTERTAINMENT, "BOOKMYSHOW": C.ENTERTAINMENT,
    "PVR": C.ENTERTAINMENT, "HOTSTAR": C.ENTERTAINMENT,
    "PHARMEASY": C.HEALTH, "NETMEDS": C.HEALTH, "PRACTO": C.HEALTH,
}

# Names written without spaces in UPI IDs ("bajajfinance@ybl") are matched as substrings of the
# space-free key, but only when the name is long enough not to collide ("OLA" is in "COLA").
_MIN_SUBSTRING_LEN = 5
_COMPACT_MERCHANTS = sorted(
    ((name.replace(" ", ""), cat, name) for name, cat in _MERCHANTS.items()
     if len(name.replace(" ", "")) >= _MIN_SUBSTRING_LEN),
    key=lambda item: -len(item[0]),
)


def _match_merchant(n: Normalised) -> RuleMatch | None:
    tokens = n.tokens
    # Pairs first, so "SWIGGY INSTAMART" wins over "SWIGGY".
    for a, b in zip(tokens, tokens[1:]):
        if (cat := _MERCHANTS.get(f"{a} {b}")) is not None:
            return RuleMatch(cat, f"merchant:{a} {b}")
    for t in tokens:
        if (cat := _MERCHANTS.get(t)) is not None:
            return RuleMatch(cat, f"merchant:{t}")
    compact = "".join(tokens)
    for name, cat, original in _COMPACT_MERCHANTS:
        if name in compact:
            return RuleMatch(cat, f"merchant:{original}")
    return None


def match_rules(n: Normalised) -> RuleMatch | None:
    if n.channel == "atm" and n.direction != "credit":
        return RuleMatch(C.CASH_WITHDRAWAL, "channel:atm")

    key = n.key
    for pattern, cat, needs_direction in _KEYWORDS:
        if needs_direction and n.direction and n.direction != needs_direction:
            continue
        if m := pattern.search(key):
            return RuleMatch(cat, f"keyword:{m.group(0)}")

    if n.direction == "credit":
        return None
    return _match_merchant(n)
