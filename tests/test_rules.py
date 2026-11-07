from txncat.categories import Category
from txncat.normalise import normalise
from txncat.rules import match_rules


def rule(description: str, direction: str | None = "debit"):
    return match_rules(normalise(description, direction))


def test_salary_credit_is_income():
    m = rule("NEFT CR-HDFC0001234-ACME TECH-SALARY NOV 2026", "credit")
    assert m.category is Category.INCOME
    assert m.reason == "keyword:SALARY"


def test_keyword_respects_direction():
    assert rule("NEFT-HDFCN1234-ACME TECH-SALARY ADVANCE", "debit") is None


def test_token_pair_beats_single_token():
    m = rule("UPI/DR/412345678901/SWIGGY INSTAMART/YESB/instamart@ybl/NA")
    assert m.category is Category.GROCERIES


def test_merchant_credit_is_not_treated_as_spending():
    assert rule("UPI/CR/412345678901/SWIGGY/YESB/swiggy@ybl/REFUND", "credit") is None


def test_atm_is_cash():
    assert rule("ATM WDL/123456/SAKET NEW DELHI").category is Category.CASH_WITHDRAWAL


def test_keywords_run_before_merchants():
    # AMAZON is a shopping merchant, but a recharge is a bill.
    assert rule("UPI/DR/1234567890/AMAZON PAY RECHARGE/YESB/amazon@apl/NA").category is (
        Category.BILLS_UTILITIES
    )


def test_short_names_match_whole_tokens_only():
    assert rule("POS 4XXXXXXXXXXX1234 COLA CORNER") is None  # contains "OLA"
    assert rule("POS 4XXXXXXXXXXX1234 OLA").category is Category.TRANSPORT


def test_names_glued_together_in_upi_ids_still_match():
    m = rule("UPI/412345678901/NA/bajajfinance@ybl/Yes Bank")
    assert m.category is Category.EMI_LOANS


def test_unknown_merchant_falls_through():
    assert rule("POS 4XXXXXXXXXXX1234 THEOBROMA") is None
