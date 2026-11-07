import pytest

from txncat.categories import Category
from txncat.ml import MLCategoriser
from txncat.normalise import normalise

TRAINING = [
    ("UPI/DR/111122223333/SWIGGY/YESB/swiggy@ybl/NA", "debit", Category.FOOD_DINING),
    ("UPI/DR/111122223334/SHARMA DHABA/YESB/sharmadhaba@okaxis/NA", "debit", Category.FOOD_DINING),
    ("POS 4XXXXXXXXXXX1234 GUPTA SWEETS", "debit", Category.FOOD_DINING),
    ("UPI/DR/222233334444/RAJ KIRANA STORE/HDFC/rajkirana@oksbi/NA", "debit", Category.GROCERIES),
    ("POS 4XXXXXXXXXXX1234 BLINKIT", "debit", Category.GROCERIES),
    ("UPI/DR/333344445555/GOEL SUPERMARKET/SBIN/goelsuper@ybl/NA", "debit", Category.GROCERIES),
    ("UPI/DR/444455556666/RAHUL VERMA/ICIC/rahul.verma@okicici/TRIP", "debit", Category.P2P_TRANSFER),
    ("UPI/CR/444455556667/PRIYA SHARMA/ICIC/priya.sharma@okaxis/GIFT", "credit", Category.P2P_TRANSFER),
    ("IMPS/P2A/555566667777/ANKIT GUPTA/HDFC/DINNER SPLIT", "debit", Category.P2P_TRANSFER),
    ("NEFT CR-HDFC0001234-ACME TECHNOLOGIES PVT LTD-SALARY NOV 2026", "credit", Category.INCOME),
    ("NEFT CR-ICIC0004321-NIMBUS LABS LLP-SAL OCT", "credit", Category.INCOME),
    ("ACH C- INFOSYS LTD DIVIDEND-1234567890", "credit", Category.INCOME),
]


@pytest.fixture(scope="session")
def trained_ml() -> MLCategoriser:
    items = [normalise(d, direction) for d, direction, _ in TRAINING]
    return MLCategoriser().fit(items, [c for _, _, c in TRAINING])
