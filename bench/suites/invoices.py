"""Invoice field extraction: a small, machine-checkable suite for local models.

Four layouts of rising difficulty, generated from a fixed seed so ground truth is exact:

* ``form``       labelled fields; a regex parser handles it with no model at all
* ``email``      prose, dates in mixed formats, currency as a symbol
* ``itemized``   no printed total: the amount due is the sum of the line amounts
* ``distractor`` two companies and two amounts (previous balance vs amount due)

Verifiers are deterministic and deliberately imperfect: they catch invented numbers, ungrounded
vendors and impossible dates, but cannot tell which of two grounded companies is the seller. That
gap is what calibration reports are for.
"""
from __future__ import annotations

import random
import re
from datetime import date

from pydantic import BaseModel, Field

from warmset import Deterministic, Model, NotApplicable, StepContext, grounded, step, verifier
from warmset.verify import Call

NAME = "invoices"
MODELS = {"small": "llama3.2:latest", "large": "qwen3.5:4b"}

VENDORS = ["Acme Corp", "Globex Ltd", "Initech", "Umbrella Supplies", "Stark Industrial", "Wayne Logistics",
           "Hooli Cloud", "Vandelay Imports", "Soylent Foods", "Tyrell Systems", "Cyberdyne Labs", "Wonka Goods"]
CUSTOMERS = ["Northwind Traders", "Contoso", "Fabrikam", "Adventure Works", "Litware", "Proseware"]
ITEMS = ["Widget", "Cable kit", "Service hour", "License seat", "Sensor", "Bracket", "Support plan"]
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "Rs."}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


class Invoice(BaseModel):
    invoice_id: str = Field(description="invoice number exactly as printed")
    date: str = Field(description="invoice date as YYYY-MM-DD")
    vendor: str = Field(description="the company that issued the invoice (the seller)")
    currency: str = Field(description="ISO 4217 code, e.g. USD")
    total: float = Field(description="amount due on this invoice")


SYSTEM = """Extract the invoice fields as JSON.
- invoice_id: exactly as printed.
- date: the invoice date as YYYY-MM-DD.
- vendor: the company that issued the invoice and is being paid, not the customer.
- currency: ISO code (USD, EUR, GBP or INR); $ is USD, € is EUR, £ is GBP, Rs. is INR.
- total: the amount due now. If no total is printed, add up the line amounts.
  Ignore previous balances and amounts already paid."""


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _make(i: int, rng: random.Random) -> dict:
    layout = ("form", "email", "itemized", "distractor")[i % 4]
    vendor, customer = rng.choice(VENDORS), rng.choice(CUSTOMERS)
    cur = rng.choice(list(CURRENCIES))
    sym = CURRENCIES[cur]
    d = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    inv = f"{rng.choice(['INV', 'BX', 'ZQ'])}-{rng.randint(1000, 99999)}"
    lines = [(rng.choice(ITEMS), rng.randint(1, 9), round(rng.uniform(5, 400), 2)) for _ in range(rng.randint(2, 4))]
    total = round(sum(q * p for _, q, p in lines), 2)
    if layout == "form":
        text = (f"INVOICE\nInvoice No: {inv}\nDate: {d.isoformat()}\nVendor: {vendor}\nBill to: {customer}\n"
                f"Currency: {cur}\nTotal: {_money(total)}\n")
    elif layout == "email":
        ds = rng.choice([f"{d.day} {MONTHS[d.month - 1]} {d.year}", f"{MONTHS[d.month - 1]} {d.day}, {d.year}"])
        text = (f"Hi {customer} team,\n\nPlease find our invoice {inv} dated {ds}. The amount due is "
                f"{sym}{_money(total)} and payment is expected within 30 days.\n\nThanks,\nAccounts, {vendor}")
    elif layout == "itemized":
        body = "\n".join(f"  {name:<14} {q} x {_money(p)} = {_money(q * p)}" for name, q, p in lines)
        text = (f"{vendor}\nInvoice {inv}   Issued: {d.strftime('%d/%m/%Y')} (DD/MM/YYYY)\nCustomer: {customer}\n"
                f"Amounts in {cur}\n{body}\n(total not printed: sum of line amounts)\n")
    else:
        prev = round(rng.uniform(50, 900), 2)
        ds = f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
        text = (f"Statement from {customer} purchasing, re: supplier {vendor}.\n"
                f"Our records show a previous balance of {sym}{_money(prev)} on an older invoice, already paid.\n"
                f"New invoice {inv} from {vendor}, issued {ds}: amount due now {sym}{_money(total)}.\n"
                f"Please pay {vendor}, not {customer}.")
    expected = {"invoice_id": inv, "date": d.isoformat(), "vendor": vendor, "currency": cur, "total": total}
    split = ("dev", "dev", "val", "test", "test")[i % 5]
    return {"id": f"inv{i:03d}", "layout": layout, "split": split, "input": {"text": text}, "expected": expected}


def load_tasks(split: str, n: int = 100, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    tasks = [_make(i, rng) for i in range(n)]
    return [t for t in tasks if split in ("all", t["split"])]


# ---------------------------------------------------------------- deterministic path

_FORM = re.compile(r"Invoice No: (?P<id>\S+)\nDate: (?P<date>\d{4}-\d{2}-\d{2})\nVendor: (?P<vendor>[^\n]+)\n"
                   r"(?:.*\n)*?Currency: (?P<cur>[A-Z]{3})\nTotal: (?P<total>[\d,]+\.\d{2})")


def parse_form(text: str) -> Invoice:
    m = _FORM.search(text)
    if not m:
        raise NotApplicable("not the labelled form layout")
    return Invoice(invoice_id=m["id"], date=m["date"], vendor=m["vendor"].strip(), currency=m["cur"],
                   total=float(m["total"].replace(",", "")))


# ---------------------------------------------------------------- verifiers

_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}")


def _amounts(text: str) -> list[float]:
    return [float(a.replace(",", "")) for a in _AMOUNT.findall(text)]


@verifier("total_supported")
def total_supported(out: Invoice, call: Call) -> object:
    """The total is printed in the text, or equals the sum of the itemised line amounts."""
    text = call.text()
    printed = _amounts(text)
    if any(abs(out.total - a) < 0.005 for a in printed):
        return True
    line_amounts = [float(m.replace(",", "")) for m in re.findall(r"= ([\d,]+\.\d{2})", text)]
    if line_amounts and abs(out.total - sum(line_amounts)) < 0.01:
        return True
    return f"total {out.total} is neither printed nor the sum of line amounts"


@verifier("date_plausible")
def date_plausible(out: Invoice, call: Call) -> object:
    try:
        d = date.fromisoformat(out.date)
    except ValueError:
        return f"not an ISO date: {out.date!r}"
    text = call.text()
    if str(d.year) not in text or not re.search(rf"(?<!\d)0?{d.day}(?!\d)", text):
        return f"year or day of {out.date} not in the text"
    return True


@verifier("currency_supported")
def currency_supported(out: Invoice, call: Call) -> object:
    sym = CURRENCIES.get(out.currency)
    text = call.text()
    if sym is None:
        return f"unknown currency {out.currency!r}"
    return out.currency in text or sym in text or f"{out.currency} not indicated in the text"


def shape(text: str) -> str:
    """Cheap input features for per-bucket pass rates. Hand-written, as a developer would: it looks at
    surface structure only (line items? several amounts? several company-like names?), never at labels."""
    if re.search(r"= [\d,]+\.\d{2}", text):
        return "line_items"
    if len(_AMOUNT.findall(text)) >= 2:
        return "multi_amount"
    return "single_amount"


@step(alternatives=[Deterministic(parse_form), Model("small"), Model("large")],
      verify=[grounded("invoice_id", case=True), grounded("vendor"), total_supported, date_plausible,
              currency_supported],
      features=shape)
async def extract_invoice(ctx: StepContext, text: str) -> Invoice:
    return await ctx.generate(system=SYSTEM, user=text, schema=Invoice, max_tokens=200)


async def pipeline(text: str) -> Invoice:
    return await extract_invoice(text)


def check(out: Invoice, expected: dict) -> bool:
    return (out.invoice_id.strip() == expected["invoice_id"]
            and out.date.strip() == expected["date"]
            and out.vendor.strip().rstrip(".").lower() == expected["vendor"].lower()
            and out.currency.strip().upper() == expected["currency"]
            and abs(out.total - expected["total"]) < 0.01)
