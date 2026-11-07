from txncat.normalise import normalise


def test_same_merchant_different_references_share_a_cache_key():
    a = normalise("UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA", "debit")
    b = normalise("UPI/DR/998877665544/SWIGGY/YESB/swiggy@ybl/NA", "debit")
    assert a.cache_key == b.cache_key


def test_strips_reference_numbers_ifsc_and_masked_cards():
    n = normalise("UPI-SWIGGY-SWIGGY@YBL-YESB0000001-412345678901-UPI", "debit")
    assert n.tokens == ("SWIGGY", "SWIGGY")
    card = normalise("POS 4XXXXXXXXXXX1234 DMART AVENUE", "debit")
    assert card.tokens == ("DMART", "AVENUE")


def test_keeps_only_the_local_part_of_a_upi_id():
    n = normalise("UPI/412345678901/Pay to merchant/gupta.kirana12@okaxis/Axis Bank", "debit")
    assert "GUPTA" in n.tokens and "KIRANA" in n.tokens
    assert not any("OKAXIS" in t for t in n.tokens)


def test_phone_number_upi_id_leaves_no_name():
    n = normalise("UPI/412345678901/NA/9876543210@ybl/HDFC Bank", "debit")
    assert n.tokens == ("HDFC",)


def test_months_and_years_are_dropped_so_monthly_salary_keeps_its_key():
    nov = normalise("NEFT CR-HDFC0001234-ACME TECH-SALARY NOV 2026", "credit")
    dec = normalise("NEFT CR-HDFC0001234-ACME TECH-SALARY DEC 2026", "credit")
    assert nov.key == dec.key == "ACME TECH SALARY"


def test_detects_channel():
    assert normalise("UPI/DR/1/X/YESB/x@ybl/NA").channel == "upi"
    assert normalise("NEFT-HDFCN1234-RAJESH KUMAR-RENT").channel == "neft"
    assert normalise("ACH D- BAJAJ FINANCE LTD-A123456789").channel == "ach"
    assert normalise("ATM WDL/123456/SAKET NEW DELHI").channel == "atm"
    assert normalise("POS 4XXXXXXXXXXX1234 CROMA").channel == "card"
    assert normalise("BBPS/AIRTEL/123456789012").channel == "billpay"
    assert normalise("SOMETHING ELSE").channel == "other"


def test_direction_falls_back_to_dr_cr_markers():
    assert normalise("UPI/DR/1/X/YESB/x@ybl/NA").direction == "debit"
    assert normalise("UPI/CR/1/X/YESB/x@ybl/NA").direction == "credit"
    assert normalise("UPI/CR/1/X/YESB/x@ybl/NA", "debit").direction == "debit"
    assert normalise("BBPS/AIRTEL/1").direction is None


def test_features_include_channel_and_direction():
    n = normalise("POS 4XXXXXXXXXXX1234 CROMA", "debit")
    assert n.features == "__card__ __debit__ CROMA"
