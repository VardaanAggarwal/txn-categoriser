"""A synthetic, labelled dataset in the formats Indian banks actually print.

No real statement data is used. Descriptions are generated from templates modelled on common
formats (UPI in three bank styles, NEFT, IMPS, card POS, NACH/ACH auto-debits, BBPS bill payments
and ATM withdrawals), with random reference numbers, UPI IDs and names. The generator knows the
true category of every row, which is what makes it usable as an evaluation set.

Its limits are documented in the README: real statements are messier, and template data is easier
than real data, so absolute accuracy here overstates real-world accuracy.
"""

import csv
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .categories import Category as C


@dataclass(frozen=True)
class Row:
    description: str
    direction: str
    category: C
    merchant_id: str


# (name, kind): kind is "brand", "local" (small shop) or "person".
_B, _L, _P = "brand", "local", "person"

_MERCHANTS: dict[C, list[tuple[str, str]]] = {
    C.FOOD_DINING: [
        ("SWIGGY", _B), ("ZOMATO", _B), ("DOMINOS", _B), ("MCDONALDS", _B), ("KFC", _B),
        ("STARBUCKS", _B), ("BURGER KING", _B), ("HALDIRAMS", _B), ("CHAAYOS", _B),
        ("BIKANERVALA", _B), ("BEHROUZ BIRYANI", _B), ("EATSURE", _B), ("BLUE TOKAI", _B),
        ("THEOBROMA", _B), ("WOW MOMO", _B), ("SAGAR RATNA", _B),
        ("SHARMA DHABA", _L), ("GUPTA SWEETS", _L), ("CAFE DELHI HEIGHTS", _L),
        ("AGGARWAL SWEET CORNER", _L), ("PUNJABI RASOI RESTAURANT", _L), ("MOMO POINT", _L),
        ("CHAI SUTTA BAR", _L), ("DEVI FOOD COURT", _L), ("BIRYANI BLUES", _L),
    ],
    C.GROCERIES: [
        ("BLINKIT", _B), ("ZEPTO", _B), ("BIGBASKET", _B), ("DMART", _B), ("SWIGGY INSTAMART", _B),
        ("JIOMART", _B), ("RELIANCE FRESH", _B), ("SPENCERS RETAIL", _B), ("NATURES BASKET", _B),
        ("MORE RETAIL", _B), ("COUNTRY DELIGHT", _B), ("MILKBASKET", _B),
        ("RAJ KIRANA STORE", _L), ("BANSAL GENERAL STORE", _L), ("MOTHER DAIRY BOOTH", _L),
        ("SHARMA PROVISION STORE", _L), ("FRESH FRUITS AND VEG", _L), ("JAIN DAIRY", _L),
        ("GOEL SUPERMARKET", _L),
    ],
    C.TRANSPORT: [
        ("UBER", _B), ("OLA", _B), ("RAPIDO", _B), ("IRCTC", _B), ("DMRC", _B), ("INDIGO", _B),
        ("AIR INDIA", _B), ("REDBUS", _B), ("NETC FASTAG", _B), ("NAMMA YATRI", _B),
        ("BLU SMART", _B), ("MAKEMYTRIP", _B),
        ("HP PETROL PUMP", _L), ("INDIAN OIL FUEL STATION", _L), ("BHARAT PETROLEUM", _L),
        ("SHELL FUEL STATION", _L), ("CITY PARKING", _L), ("SHARMA AUTO SERVICE", _L),
    ],
    C.SHOPPING: [
        ("AMAZON", _B), ("FLIPKART", _B), ("MYNTRA", _B), ("AJIO", _B), ("NYKAA", _B),
        ("MEESHO", _B), ("CROMA", _B), ("RELIANCE DIGITAL", _B), ("DECATHLON", _B), ("IKEA", _B),
        ("TATA CLIQ", _B), ("LENSKART", _B), ("UNIQLO", _B), ("ZARA", _B),
        ("KAPOOR GARMENTS", _L), ("SHREE FOOTWEAR", _L), ("MOBILE POINT ELECTRONICS", _L),
        ("SINGH CLOTH HOUSE", _L), ("STYLE BAZAAR", _L), ("VERMA GIFT GALLERY", _L),
    ],
    C.BILLS_UTILITIES: [
        ("BSES RAJDHANI", _B), ("BSES YAMUNA", _B), ("TATA POWER DDL", _B), ("AIRTEL", _B),
        ("JIO", _B), ("VI", _B), ("ACT FIBERNET", _B), ("INDRAPRASTHA GAS", _B),
        ("DELHI JAL BOARD", _B), ("TATA PLAY", _B), ("EXCITEL BROADBAND", _B),
        ("AIRTEL POSTPAID", _B), ("JIO PREPAID RECHARGE", _B),
    ],
    C.RENT: [
        ("RAJESH KUMAR", _P), ("SUNITA MALHOTRA", _P), ("ARVIND KHANNA", _P),
        ("MEENA KAPOOR", _P), ("SURESH YADAV", _P), ("KAMLESH ARORA", _P), ("VIJAY BHATIA", _P),
        ("NIRMALA SETHI", _P), ("NOBROKER", _B), ("CRED RENTPAY", _B), ("HOUSING COM", _B),
    ],
    C.EMI_LOANS: [
        ("BAJAJ FINANCE LTD", _B), ("TATA CAPITAL", _B), ("HOME CREDIT INDIA", _B),
        ("HDFC BANK LOAN", _B), ("ICICI BANK LOAN EMI", _B), ("KREDITBEE", _B), ("SLICE", _B),
        ("CRED CLUB", _B), ("IDFC FIRST BANK LOAN", _B), ("SIMPL", _B), ("LAZYPAY", _B),
    ],
    C.INVESTMENTS: [
        ("ZERODHA BROKING", _B), ("GROWW", _B), ("UPSTOX", _B), ("KUVERA", _B), ("ANGEL ONE", _B),
        ("INDMONEY", _B), ("ET MONEY", _B), ("PAYTM MONEY", _B), ("NPS TRUST", _B),
        ("ICICI PRUDENTIAL MF", _B), ("HDFC MUTUAL FUND", _B), ("NIPPON INDIA MF", _B),
        ("BSE STAR MF SIP", _B),
    ],
    C.P2P_TRANSFER: [
        ("RAHUL VERMA", _P), ("PRIYA SHARMA", _P), ("ANKIT GUPTA", _P), ("NEHA SINGH", _P),
        ("MOHIT JAIN", _P), ("POOJA MEHTA", _P), ("KARAN MALHOTRA", _P), ("SNEHA REDDY", _P),
        ("ARJUN NAIR", _P), ("DIVYA IYER", _P), ("ROHAN DAS", _P), ("ISHA BANSAL", _P),
        ("AMIT TIWARI", _P), ("KAVYA PILLAI", _P), ("VARUN CHOPRA", _P), ("RIYA SAXENA", _P),
        ("SAURABH MISHRA", _P), ("TANVI JOSHI", _P), ("NIKHIL RAO", _P), ("SIMRAN KAUR", _P),
    ],
    C.INCOME: [
        ("ACME TECHNOLOGIES PVT LTD", "employer"), ("BRIGHTPATH SOFTWARE PVT LTD", "employer"),
        ("NIMBUS LABS LLP", "employer"), ("ORBIT SYSTEMS PVT LTD", "employer"),
        ("INTEREST", "interest"),
        ("INFOSYS LTD", "dividend"), ("ITC LTD", "dividend"), ("HDFC BANK LTD", "dividend"),
        ("TCS LTD", "dividend"), ("COAL INDIA LTD", "dividend"),
    ],
    C.ENTERTAINMENT: [
        ("NETFLIX", _B), ("SPOTIFY", _B), ("BOOKMYSHOW", _B), ("PVR", _B), ("HOTSTAR", _B),
        ("PRIME VIDEO", _B), ("YOUTUBE PREMIUM", _B), ("SONYLIV", _B), ("ZEE5", _B), ("INOX", _B),
        ("CINEPOLIS", _B), ("SMAAASH", _B), ("FUN CITY", _B), ("DISTRICT APP", _B),
    ],
    C.HEALTH: [
        ("APOLLO PHARMACY", _B), ("PHARMEASY", _B), ("NETMEDS", _B), ("PRACTO", _B),
        ("TATA 1MG", _B), ("CULT FIT", _B), ("MAX HEALTHCARE", _B), ("FORTIS", _B),
        ("DR LAL PATHLABS", _B), ("MEDPLUS", _B),
        ("SHARMA MEDICAL STORE", _L), ("CITY CLINIC", _L), ("JAIN MEDICOS", _L),
        ("LIFELINE DIAGNOSTICS", _L), ("SMILE DENTAL CARE", _L), ("GUPTA CHEMIST", _L),
    ],
    C.CASH_WITHDRAWAL: [
        ("CONNAUGHT PLACE", "atm"), ("SECTOR 18 NOIDA", "atm"), ("MG ROAD", "atm"),
        ("LAJPAT NAGAR", "atm"), ("KAROL BAGH", "atm"), ("DLF PHASE 2", "atm"),
        ("RAJOURI GARDEN", "atm"), ("SAKET", "atm"),
    ],
}

# (channel, weight) per category. Directions are debit unless listed in _CREDIT_SHARE.
_CHANNELS: dict[C, list[tuple[str, float]]] = {
    C.FOOD_DINING: [("upi", 0.75), ("card", 0.25)],
    C.GROCERIES: [("upi", 0.7), ("card", 0.3)],
    C.TRANSPORT: [("upi", 0.65), ("card", 0.35)],
    C.SHOPPING: [("upi", 0.5), ("card", 0.5)],
    C.BILLS_UTILITIES: [("bill", 0.5), ("upi", 0.5)],
    C.RENT: [("neft", 0.4), ("imps", 0.2), ("upi", 0.4)],
    C.EMI_LOANS: [("ach", 0.7), ("upi", 0.3)],
    C.INVESTMENTS: [("ach", 0.6), ("upi", 0.4)],
    C.P2P_TRANSFER: [("upi", 0.75), ("imps", 0.25)],
    C.ENTERTAINMENT: [("upi", 0.5), ("card", 0.5)],
    C.HEALTH: [("upi", 0.7), ("card", 0.3)],
}

_COUNTS: dict[C, int] = {
    C.FOOD_DINING: 150, C.GROCERIES: 130, C.TRANSPORT: 120, C.SHOPPING: 120,
    C.BILLS_UTILITIES: 100, C.RENT: 70, C.EMI_LOANS: 80, C.INVESTMENTS: 80,
    C.P2P_TRANSFER: 150, C.INCOME: 80, C.ENTERTAINMENT: 90, C.HEALTH: 90,
    C.CASH_WITHDRAWAL: 60,
}

_BANK_CODES = ["YESB", "ICIC", "HDFC", "SBIN", "UTIB", "KKBK"]
_BANK_NAMES = ["Yes Bank", "ICICI Bank", "Axis Bank", "HDFC Bank", "State Bank Of India",
               "Kotak Mahindra Bank"]
_BRAND_HANDLES = ["ybl", "axl", "icici", "hdfcbank", "paytm", "yesbank"]
_PERSON_HANDLES = ["okaxis", "okhdfcbank", "oksbi", "okicici", "ybl", "paytm", "ibl"]
_UPI_REMARKS = ["Payment from Phone", "UPI", "Paid via app", "NA", "Pay to merchant"]
_P2P_REMARKS = ["DINNER SPLIT", "TRIP", "SENT USING PAYTM", "NA", "GIFT", "PAID BACK", "CAB SHARE"]
_RENT_REMARKS = ["RENT", "HOUSE RENT", "FLAT RENT", "RENT NOV", "RENT OCT"]
_CITIES = ["NEW DELHI", "GURGAON", "NOIDA", "BANGALORE", "MUMBAI"]
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _ifsc(rng: random.Random) -> str:
    return rng.choice(_BANK_CODES) + "0" + _digits(rng, 6)


def _masked_card(rng: random.Random) -> str:
    return rng.choice("45") + "X" * 11 + _digits(rng, 4)


def _vpa(rng: random.Random, name: str, kind: str) -> str:
    words = "".join(ch for ch in name.lower() if ch.isalnum() or ch == " ").split()
    if kind == _B:
        local = rng.choice(["".join(words), ".".join(words)])
        if rng.random() < 0.3:
            local += _digits(rng, 1)
        return f"{local}@{rng.choice(_BRAND_HANDLES)}"
    if kind == _L:
        if rng.random() < 0.2:  # a generic QR-code UPI ID that says nothing about the shop
            return f"paytmqr{_digits(rng, 10)}@paytm"
        return f"{''.join(words[:2])}{_digits(rng, rng.choice([0, 2, 3]))}@{rng.choice(_PERSON_HANDLES)}"
    roll = rng.random()
    if roll < 0.25:  # a phone-number UPI ID: no name at all
        return f"{rng.choice('6789')}{_digits(rng, 9)}@{rng.choice(_PERSON_HANDLES)}"
    if roll < 0.6:
        return f"{'.'.join(words)}@{rng.choice(_PERSON_HANDLES)}"
    return f"{words[0]}{_digits(rng, rng.choice([2, 3, 4]))}@{rng.choice(_PERSON_HANDLES)}"


def _upi(rng: random.Random, name: str, kind: str, direction: str, remark: str | None) -> str:
    ref = _digits(rng, 12)
    vpa = _vpa(rng, name, kind)
    remark = remark or rng.choice(_UPI_REMARKS)
    style = rng.randrange(3)
    if style == 0:  # banks often truncate the payee name to ~20 characters
        return f"UPI/{'DR' if direction == 'debit' else 'CR'}/{ref}/{name[:20]}/{rng.choice(_BANK_CODES)}/{vpa}/{remark}"
    if style == 1:
        return f"UPI-{name}-{vpa.upper()}-{_ifsc(rng)}-{ref}-{remark.upper()}"
    return f"UPI/{ref}/{remark}/{vpa}/{rng.choice(_BANK_NAMES)}"  # payee only in the UPI ID


def _card(rng: random.Random, name: str) -> str:
    if rng.random() < 0.5:
        return f"POS {_masked_card(rng)} {name}"
    return f"PCD/{_masked_card(rng)}/{name}/{rng.choice(_CITIES)}"


def _neft(rng: random.Random, name: str, direction: str, remark: str) -> str:
    if direction == "credit":
        return f"NEFT CR-{_ifsc(rng)}-{name}-{remark}"
    return f"NEFT-{rng.choice(_BANK_CODES)}N{_digits(rng, 14)}-{name}-{remark}-{_ifsc(rng)}"


def _imps(rng: random.Random, name: str, remark: str) -> str:
    return f"IMPS/P2A/{_digits(rng, 12)}/{name}/{rng.choice(_BANK_CODES)}/{remark}"


def _ach(rng: random.Random, name: str, direction: str) -> str:
    if direction == "credit":
        return f"ACH C- {name}-{_digits(rng, 10)}"
    if rng.random() < 0.5:
        return f"ACH D- {name}-{rng.choice('ABCDEFGH')}{_digits(rng, 12)}"
    return f"NACH DR {name} {_digits(rng, 10)}"


def _bill(rng: random.Random, name: str) -> str:
    if rng.random() < 0.5:
        return f"BIL/ONL/{_digits(rng, 9)}/{name}/{_digits(rng, 8)}"
    return f"BBPS/{name}/{_digits(rng, 12)}"


def _atm(rng: random.Random, location: str) -> str:
    if rng.random() < 0.5:
        return f"ATM WDL/{_digits(rng, 6)}/{location} {rng.choice(_CITIES)}"
    return f"NWD-{_masked_card(rng)}-S1AN{_digits(rng, 6)}-{location}"


def _income(rng: random.Random, name: str, kind: str) -> str:
    if kind == "employer":
        month, year = rng.choice(_MONTHS), rng.choice(["2025", "2026"])
        remark = rng.choice([f"SALARY {month} {year}", f"SAL {month}", f"SALARY FOR {month}"])
        return _neft(rng, name, "credit", remark)
    if kind == "interest":
        return f"INT.PD:{_digits(rng, 12)}:01-{_digits(rng, 2)}-2026 TO 30-{_digits(rng, 2)}-2026"
    return rng.choice([f"ACH C- {name} DIVIDEND-{_digits(rng, 10)}", f"{name} DIV {_digits(rng, 8)}"])


def _describe(rng: random.Random, category: C, name: str, kind: str) -> tuple[str, str]:
    """One (description, direction) pair for a merchant in a category."""
    if category is C.CASH_WITHDRAWAL:
        return _atm(rng, name), "debit"
    if category is C.INCOME:
        return _income(rng, name, kind), "credit"

    direction = "debit"
    remark: str | None = None
    if category is C.P2P_TRANSFER:
        direction = rng.choice(["debit", "credit"])
        remark = rng.choice(_P2P_REMARKS)
    elif category is C.RENT and kind == _P:
        # Rent to a landlord is often sent with no remark, which makes it look like any other
        # transfer to a person. That's a real, hard case, so it stays in the data.
        remark = rng.choice(_RENT_REMARKS) if rng.random() < 0.55 else rng.choice(["NA", "TRANSFER", "MONTHLY"])

    channels, weights = zip(*_CHANNELS[category])
    channel = rng.choices(channels, weights)[0]
    if channel == "upi":
        return _upi(rng, name, kind, direction, remark), direction
    if channel == "card":
        return _card(rng, name), direction
    if channel == "neft":
        return _neft(rng, name, direction, remark or "NA"), direction
    if channel == "imps":
        return _imps(rng, name, remark or "NA"), direction
    if channel == "ach":
        return _ach(rng, name, direction), direction
    return _bill(rng, name), direction


def generate(seed: int = 42) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    for category, count in _COUNTS.items():
        merchants = _MERCHANTS[category]
        for _ in range(count):
            name, kind = rng.choice(merchants)
            description, direction = _describe(rng, category, name, kind)
            rows.append(Row(description, direction, category, f"{category.value}:{name}"))
    rng.shuffle(rows)
    return rows


def write_csv(rows: Sequence[Row], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "direction", "category", "merchant_id"])
        for r in rows:
            writer.writerow([r.description, r.direction, r.category.value, r.merchant_id])


def load_csv(path: str | Path) -> list[Row]:
    with open(path, newline="") as f:
        return [
            Row(r["description"], r["direction"], C(r["category"]), r["merchant_id"])
            for r in csv.DictReader(f)
        ]


def merchant_disjoint_split(
    rows: Sequence[Row], test_fraction: float = 0.3, seed: int = 7
) -> tuple[list[Row], list[Row]]:
    """Split so no merchant or person appears in both train and test.

    A random row split would put the same merchant on both sides and mostly measure memorisation.
    This split asks the harder question: how well does it handle merchants it has never seen?
    Merchants are split within each category, so every category has some in train and in test.
    """
    rng = random.Random(seed)
    test_merchants: set[str] = set()
    for category in sorted({r.category for r in rows}):
        merchants = sorted({r.merchant_id for r in rows if r.category is category})
        rng.shuffle(merchants)
        n_test = min(len(merchants) - 1, max(1, round(len(merchants) * test_fraction)))
        test_merchants.update(merchants[:n_test])
    train = [r for r in rows if r.merchant_id not in test_merchants]
    test = [r for r in rows if r.merchant_id in test_merchants]
    return train, test


def random_split(
    rows: Sequence[Row], test_fraction: float = 0.3, seed: int = 7
) -> tuple[list[Row], list[Row]]:
    """Plain random split, where test merchants have usually been seen in training."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]
