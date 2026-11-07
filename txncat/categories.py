"""The fixed set of spending categories shared by every layer."""

from enum import StrEnum


class Category(StrEnum):
    FOOD_DINING = "food_dining"
    GROCERIES = "groceries"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    BILLS_UTILITIES = "bills_utilities"
    RENT = "rent"
    EMI_LOANS = "emi_loans"
    INVESTMENTS = "investments"
    P2P_TRANSFER = "p2p_transfer"
    INCOME = "income"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    CASH_WITHDRAWAL = "cash_withdrawal"
    OTHER = "other"


DESCRIPTIONS: dict[Category, str] = {
    Category.FOOD_DINING: "restaurants, cafes, sweet shops and food delivery",
    Category.GROCERIES: "supermarkets, kirana stores, dairy and quick-commerce grocery apps",
    Category.TRANSPORT: "cabs, autos, metro, trains, flights, fuel, parking and FASTag",
    Category.SHOPPING: "e-commerce, clothing, footwear, electronics and retail stores",
    Category.BILLS_UTILITIES: "electricity, water, gas, mobile recharge, broadband and DTH",
    Category.RENT: "house rent paid to a landlord or through a rent-payment app",
    Category.EMI_LOANS: "loan EMIs, buy-now-pay-later and credit card bill payments",
    Category.INVESTMENTS: "stocks, mutual funds, SIPs, NPS and brokerage accounts",
    Category.P2P_TRANSFER: "money sent to or received from a friend or family member",
    Category.INCOME: "salary, interest and dividends received",
    Category.ENTERTAINMENT: "streaming subscriptions, movies, events and gaming",
    Category.HEALTH: "pharmacies, doctors, hospitals, diagnostics and fitness",
    Category.CASH_WITHDRAWAL: "ATM cash withdrawals",
    Category.OTHER: "anything that does not fit the categories above",
}
