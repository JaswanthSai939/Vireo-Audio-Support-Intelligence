from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# VIREO AUDIO - STEP 5
# BUSINESS GOAL + FINANCIAL IMPACT ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS FROM SUPPORT POLICY / PREVIOUS ANALYSIS
# ============================================================

BLENDED_CONTACT_COST = 290
SLA_BREACH_CREDIT = 350
TRANSFER_COST = 305

REPLACEMENT_LOGISTICS_COST = 340

# Dataset covers Jan 2025 - Jun 2026 = 18 months.
# For quarterly normalization:
DATASET_MONTHS = 18
DATASET_QUARTERS = DATASET_MONTHS / 3


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"₹{value:,.0f}"


def percentage(value):
    return f"{value:.2f}%"


def safe_float(value):
    if pd.isna(value):
        return 0.0
    return float(value)


def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    return pd.read_csv(path)


def load_json(filename):
    path = REPORTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing report: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# LOAD TICKETS
# ============================================================

def prepare_tickets():

    tickets = load_csv("tickets.csv")

    raw_count = len(tickets)

    # Parse dates.
    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce"
    )

    tickets["resolved_at"] = pd.to_datetime(
        tickets["resolved_at"],
        errors="coerce"
    )

    # Prefer helpdesk over legacy_fd.
    if "source_system" in tickets.columns:

        tickets["_source_priority"] = (
            tickets["source_system"]
            .astype(str)
            .str.lower()
            .eq("helpdesk")
            .astype(int)
        )

    else:

        tickets["_source_priority"] = 0

    tickets["_completeness"] = (
        tickets.notna().sum(axis=1)
    )

    tickets = tickets.sort_values(
        [
            "ticket_id",
            "_source_priority",
            "_completeness"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )

    tickets = tickets.drop_duplicates(
        subset="ticket_id",
        keep="first"
    ).copy()

    tickets.reset_index(
        drop=True,
        inplace=True
    )

    tickets.drop(
        columns=[
            "_source_priority",
            "_completeness"
        ],
        inplace=True,
        errors="ignore"
    )

    return tickets, raw_count


# ============================================================
# REFUND ANALYSIS
# ============================================================

def analyze_refunds(tickets):

    refunds = pd.to_numeric(
        tickets["refund_amount_inr"],
        errors="coerce"
    ).fillna(0)

    refund_mask = refunds > 0

    refund_tickets = int(
        refund_mask.sum()
    )

    refund_amount = float(
        refunds.sum()
    )

    refund_rate = (
        refund_tickets /
        len(tickets) *
        100
    )

    # Refunds by category.
    category_table = (
        tickets.loc[refund_mask]
        .assign(
            refund_amount=refunds[refund_mask]
        )
        .groupby("category")
        .agg(
            tickets=("ticket_id", "count"),
            refund_amount=("refund_amount", "sum")
        )
        .sort_values(
            "refund_amount",
            ascending=False
        )
    )

    # Refunds by product.
    product_table = (
        tickets.loc[refund_mask]
        .assign(
            refund_amount=refunds[refund_mask]
        )
        .groupby("product_sku")
        .agg(
            tickets=("ticket_id", "count"),
            refund_amount=("refund_amount", "sum")
        )
        .sort_values(
            "refund_amount",
            ascending=False
        )
    )

    return {
        "ticket_count": refund_tickets,
        "amount": refund_amount,
        "rate": refund_rate,
        "by_category": category_table,
        "by_product": product_table,
    }


# ============================================================
# REPLACEMENT ANALYSIS
# ============================================================

def analyze_replacements(tickets):

    products = load_csv("products.csv")

    replacement_mask = (
        tickets["replacement_issued"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("Y")
    )

    replacement_count = int(
        replacement_mask.sum()
    )

    products["unit_cost_inr"] = pd.to_numeric(
        products["unit_cost_inr"],
        errors="coerce"
    )

    product_costs = products[
        [
            "sku",
            "unit_cost_inr"
        ]
    ].drop_duplicates(
        subset=["sku"]
    )

    merged = tickets.merge(
        product_costs,
        left_on="product_sku",
        right_on="sku",
        how="left"
    )

    valid_mask = (
        replacement_mask
        & merged["unit_cost_inr"].notna()
    )

    estimated_cost = float(
        (
            merged.loc[
                valid_mask,
                "unit_cost_inr"
            ]
            + REPLACEMENT_LOGISTICS_COST
        ).sum()
    )

    missing_cost = int(
        (
            replacement_mask
            & merged["unit_cost_inr"].isna()
        ).sum()
    )

    return {
        "ticket_count": replacement_count,
        "estimated_cost": estimated_cost,
        "rate": (
            replacement_count /
            len(tickets) *
            100
        ),
        "missing_product_cost": missing_cost,
    }


# ============================================================
# SLA ANALYSIS
# ============================================================

def analyze_sla(tickets):

    sla_targets = {
        "chat": 15,
        "voice": 120,
        "social": 240,
        "email": 480,
    }

    tickets = tickets.copy()

    tickets["response_minutes"] = (
        (
            tickets["first_response_at"]
            if "first_response_at" in tickets.columns
            else pd.NaT
        )
    )

    if "first_response_at" in tickets.columns:

        tickets["first_response_at"] = pd.to_datetime(
            tickets["first_response_at"],
            errors="coerce"
        )

        tickets["response_minutes"] = (
            tickets["first_response_at"]
            - tickets["created_at"]
        ).dt.total_seconds() / 60

    tickets["sla_target"] = (
        tickets["channel"]
        .astype(str)
        .str.lower()
        .map(sla_targets)
    )

    breach_mask = (
        tickets["response_minutes"].notna()
        & tickets["sla_target"].notna()
        & (
            tickets["response_minutes"]
            > tickets["sla_target"]
        )
    )

    breach_count = int(
        breach_mask.sum()
    )

    breach_rate = (
        breach_count /
        len(tickets) *
        100
    )

    breach_cost = (
        breach_count *
        SLA_BREACH_CREDIT
    )

    return {
        "count": breach_count,
        "rate": breach_rate,
        "cost": float(breach_cost),
    }


# ============================================================
# TRANSFER ANALYSIS
# ============================================================

def analyze_transfers(tickets):

    transfers = pd.to_numeric(
        tickets["transfers"],
        errors="coerce"
    ).fillna(0)

    total_transfers = int(
        transfers.sum()
    )

    tickets_with_transfer = int(
        (transfers > 0).sum()
    )

    transfer_rate = (
        tickets_with_transfer /
        len(tickets) *
        100
    )

    transfer_cost = (
        total_transfers *
        TRANSFER_COST
    )

    return {
        "total_transfers": total_transfers,
        "tickets_with_transfer": tickets_with_transfer,
        "rate": transfer_rate,
        "cost": float(transfer_cost),
    }


# ============================================================
# CSAT ANALYSIS
# ============================================================

def analyze_csat(tickets):

    csat = pd.to_numeric(
        tickets["csat_score"],
        errors="coerce"
    )

    # Policy/data convention:
    # 0 means no response.
    csat = csat.mask(
        csat == 0
    )

    valid = csat.dropna()

    low = int(
        (valid <= 2).sum()
    )

    high = int(
        (valid >= 4).sum()
    )

    return {
        "responses": int(len(valid)),
        "average": (
            float(valid.mean())
            if len(valid)
            else None
        ),
        "low_count": low,
        "high_count": high,
    }


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def analyze_categories(tickets):

    result = (
        tickets
        .groupby("category")
        .agg(
            tickets=("ticket_id", "count"),
            refund_tickets=(
                "refund_amount_inr",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce"
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                )
            ),
            replacement_tickets=(
                "replacement_issued",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("Y")
                    .sum()
                )
            ),
            transfers=(
                "transfers",
                "sum"
            ),
        )
        .sort_values(
            "tickets",
            ascending=False
        )
    )

    result["share_percent"] = (
        result["tickets"] /
        len(tickets) *
        100
    )

    return result


# ============================================================
# QUARTERLY NORMALIZATION
# ============================================================

def quarterly_value(
    amount
):

    return (
        amount /
        DATASET_QUARTERS
    )


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

def savings_scenarios(
    observed_cost
):

    scenarios = []

    for reduction in [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]:

        quarterly_cost = quarterly_value(
            observed_cost
        )

        savings = (
            quarterly_cost *
            reduction
        )

        scenarios.append({
            "reduction_percent": reduction * 100,
            "quarterly_savings": savings,
            "annualized_savings": savings * 4,
        })

    return scenarios


# ============================================================
# BUSINESS GOAL CANDIDATES
# ============================================================

def build_goal_candidates(
    refunds,
    replacements,
    sla,
    transfers
):

    candidates = []

    opportunities = [
        (
            "Refund leakage",
            refunds["amount"],
            "Reduce refund amount"
        ),
        (
            "Replacement cost",
            replacements["estimated_cost"],
            "Reduce replacement volume/cost"
        ),
        (
            "SLA breach credits",
            sla["cost"],
            "Reduce first-response SLA breaches"
        ),
        (
            "Internal transfer cost",
            transfers["cost"],
            "Reduce internal hand-offs"
        ),
    ]

    for name, observed_cost, description in opportunities:

        candidates.append({
            "opportunity": name,
            "description": description,
            "observed_18_month_cost": float(
                observed_cost
            ),
            "observed_quarterly_cost": float(
                quarterly_value(observed_cost)
            ),
            "10_percent_savings": float(
                quarterly_value(observed_cost)
                * 0.10
            ),
            "15_percent_savings": float(
                quarterly_value(observed_cost)
                * 0.15
            ),
            "20_percent_savings": float(
                quarterly_value(observed_cost)
                * 0.20
            ),
            "25_percent_savings": float(
                quarterly_value(observed_cost)
                * 0.25
            ),
        })

    return pd.DataFrame(
        candidates
    ).sort_values(
        "observed_18_month_cost",
        ascending=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VIREO AUDIO - STEP 5: BUSINESS GOAL ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\n[1/7] Loading ticket data...")

    tickets, raw_count = prepare_tickets()

    print(
        f"  Raw rows: {raw_count:,}"
    )

    print(
        f"  Unique tickets: {len(tickets):,}"
    )

    # --------------------------------------------------------
    # Analyze financial opportunities
    # --------------------------------------------------------

    print("\n[2/7] Analyzing refunds...")

    refunds = analyze_refunds(
        tickets
    )

    print(
        f"  Refund tickets: "
        f"{refunds['ticket_count']:,}"
    )

    print(
        f"  Refund amount: "
        f"{money(refunds['amount'])}"
    )

    print("\n[3/7] Analyzing replacements...")

    replacements = analyze_replacements(
        tickets
    )

    print(
        f"  Replacement tickets: "
        f"{replacements['ticket_count']:,}"
    )

    print(
        f"  Estimated replacement cost: "
        f"{money(replacements['estimated_cost'])}"
    )

    print("\n[4/7] Analyzing SLA and transfers...")

    sla = analyze_sla(
        tickets
    )

    transfers = analyze_transfers(
        tickets
    )

    print(
        f"  SLA breaches: "
        f"{sla['count']:,}"
    )

    print(
        f"  SLA credit exposure: "
        f"{money(sla['cost'])}"
    )

    print(
        f"  Transfer tickets: "
        f"{transfers['tickets_with_transfer']:,}"
    )

    print(
        f"  Transfer cost: "
        f"{money(transfers['cost'])}"
    )

    # --------------------------------------------------------
    # CSAT
    # --------------------------------------------------------

    print("\n[5/7] Analyzing CSAT...")

    csat = analyze_csat(
        tickets
    )

    print(
        f"  CSAT responses: "
        f"{csat['responses']:,}"
    )

    if csat["average"] is not None:

        print(
            f"  Average CSAT: "
            f"{csat['average']:.2f}"
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    print("\n[6/7] Finding major complaint categories...")

    categories = analyze_categories(
        tickets
    )

    print("\nTop categories by volume:")

    for category, row in categories.head(10).iterrows():

        print(
            f"  {str(category):<25}"
            f"{int(row['tickets']):>6} tickets "
            f"({row['share_percent']:.2f}%)"
        )

    # --------------------------------------------------------
    # Goal candidates
    # --------------------------------------------------------

    print("\n[7/7] Building business goal scenarios...")

    goal_candidates = build_goal_candidates(
        refunds=refunds,
        replacements=replacements,
        sla=sla,
        transfers=transfers,
    )

    print("\nPotential quarterly value at 15% improvement:")

    for _, row in goal_candidates.iterrows():

        print(
            f"  {row['opportunity']:<25}"
            f"{money(row['15_percent_savings'])}"
        )

    # --------------------------------------------------------
    # Repeat validation evidence
    # --------------------------------------------------------

    repeat_validation = {
        "sample_size": 50,
        "true": 33,
        "false": 17,
        "unclear": 0,
        "precision_percent": 66.0,
        "error_rate_percent": 34.0,
    }

    # --------------------------------------------------------
    # Identify leading candidate
    # --------------------------------------------------------

    # IMPORTANT:
    # We identify the largest observed financial opportunity,
    # but we do NOT claim that reducing it by 15% is guaranteed.
    #
    # The final target must be supported by the tool and validation.

    leading = goal_candidates.iloc[0]

    leading_observed = (
        float(
            leading[
                "observed_18_month_cost"
            ]
        )
    )

    leading_quarterly = (
        float(
            leading[
                "observed_quarterly_cost"
            ]
        )
    )

    leading_15 = (
        float(
            leading[
                "15_percent_savings"
            ]
        )
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = {

        "population": {
            "raw_rows": int(raw_count),
            "unique_tickets": int(len(tickets)),
            "dataset_months": DATASET_MONTHS,
            "dataset_quarters": DATASET_QUARTERS,
        },

        "financial_opportunities": {

            "refunds": {
                "ticket_count": refunds["ticket_count"],
                "rate_percent": refunds["rate"],
                "observed_18_month_amount": refunds["amount"],
                "observed_quarterly_amount": quarterly_value(
                    refunds["amount"]
                ),
            },

            "replacements": {
                "ticket_count": replacements["ticket_count"],
                "rate_percent": replacements["rate"],
                "observed_18_month_cost": replacements[
                    "estimated_cost"
                ],
                "observed_quarterly_cost": quarterly_value(
                    replacements["estimated_cost"]
                ),
            },

            "sla": sla,

            "transfers": transfers,
        },

        "csat": csat,

        "repeat_detector_validation": repeat_validation,

        "leading_financial_opportunity": {
            "opportunity": leading["opportunity"],
            "observed_18_month_cost": leading_observed,
            "observed_quarterly_cost": leading_quarterly,
            "illustrative_15_percent_quarterly_savings": leading_15,
        },

        "important_note": (
            "The 15% savings figure is an illustrative scenario, "
            "not a guaranteed business outcome."
        ),
    }

    json_path = (
        REPORTS_DIR /
        "business_goal.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    md_path = (
        REPORTS_DIR /
        "05_business_goal.md"
    )

    md = f"""# Vireo Audio — Business Goal Analysis

## 1. Population

- Raw ticket rows: **{raw_count:,}**
- Unique analytical tickets: **{len(tickets):,}**
- Historical period: **18 months**
- Normalization period: **6 quarters**

---

## 2. Observed Financial Opportunities

| Opportunity | Observed 18-month amount | Approx. quarterly amount |
|---|---:|---:|
| Refunds | {money(refunds['amount'])} | {money(quarterly_value(refunds['amount']))} |
| Replacements | {money(replacements['estimated_cost'])} | {money(quarterly_value(replacements['estimated_cost']))} |
| SLA breach credits | {money(sla['cost'])} | {money(quarterly_value(sla['cost']))} |
| Internal transfers | {money(transfers['cost'])} | {money(quarterly_value(transfers['cost']))} |

These are historical observed amounts or policy-based cost estimates.
They are not savings.

---

## 3. Volume and Rates

### Refunds

- Refund tickets: **{refunds['ticket_count']:,}**
- Refund rate: **{refunds['rate']:.2f}%**
- Refund amount: **{money(refunds['amount'])}**

### Replacements

- Replacement tickets: **{replacements['ticket_count']:,}**
- Replacement rate: **{replacements['rate']:.2f}%**
- Estimated replacement cost: **{money(replacements['estimated_cost'])}**

### SLA

- SLA breaches: **{sla['count']:,}**
- SLA breach rate: **{sla['rate']:.2f}%**
- Credit exposure: **{money(sla['cost'])}**

### Transfers

- Tickets with transfers: **{transfers['tickets_with_transfer']:,}**
- Transfer rate: **{transfers['rate']:.2f}%**
- Estimated transfer cost: **{money(transfers['cost'])}**

---

## 4. Customer Experience

- Valid CSAT responses: **{csat['responses']:,}**
- Average CSAT: **{
        f"{csat['average']:.2f}"
        if csat['average'] is not None
        else "N/A"
    }**
- Low CSAT responses (1–2): **{csat['low_count']:,}**
- High CSAT responses (4–5): **{csat['high_count']:,}**

---

## 5. Repeat-Contact Evidence

The repeat detector was validated on a sample of **50 candidate cases**.

| Validation result | Count |
|---|---:|
| TRUE | 33 |
| FALSE | 17 |
| UNCLEAR | 0 |
| Precision | 66.00% |
| Error rate | 34.00% |

Therefore the detected **3.00% strong-repeat rate should not be
treated as a confirmed KPI**.

The repeat detector should be used as a supporting signal rather
than the sole basis for the business target.

---

## 6. Largest Observed Financial Opportunity

The largest observed financial category in the available data is:

**{leading['opportunity']}**

Observed 18-month amount:

**{money(leading_observed)}**

Normalized quarterly amount:

**{money(leading_quarterly)}**

An illustrative 15% reduction would correspond to:

**{money(leading_15)} per quarter**

This is a scenario, not a guaranteed saving.

---

## 7. Potential Business Goal

A final business goal should satisfy three conditions:

1. It is measurable from the ticket data.
2. The AI-assisted tool can directly influence the underlying
   operational process.
3. The expected improvement can be validated after deployment.

The financial analysis identifies **{leading['opportunity']}**
as the largest observed cost opportunity, but the final target
should only claim savings that can be linked to the actual tool.

---

## 8. Recommended Measurement Structure

The final tool should report:

- Weekly complaint volume
- Top complaint themes
- Week-over-week change
- Refund/replacement/transfer association
- CSAT for each major theme
- Supporting examples
- Repeat-contact signal where confidence is adequate
- Estimated financial exposure

This allows management to act on the weekly digest rather than
simply receiving a list of complaints.

---

## 9. Important Limitation

The historical data shows financial exposure, not guaranteed
avoidable cost.

For example, a refund is not automatically a mistake or a cost
that can be eliminated.

Therefore the final submission should distinguish:

**Observed cost → addressable opportunity → targeted reduction
→ estimated savings**

rather than claiming that the entire historical amount is
recoverable.
"""

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(md)

    # --------------------------------------------------------
    # Save category analysis
    # --------------------------------------------------------

    category_path = (
        REPORTS_DIR /
        "business_goal_categories.csv"
    )

    categories.to_csv(
        category_path
    )

    # --------------------------------------------------------
    # Save goal candidates
    # --------------------------------------------------------

    goal_path = (
        REPORTS_DIR /
        "business_goal_candidates.csv"
    )

    goal_candidates.to_csv(
        goal_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5 BUSINESS GOAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\n✓ Created: {json_path}"
    )

    print(
        f"✓ Created: {md_path}"
    )

    print(
        f"✓ Created: {category_path}"
    )

    print(
        f"✓ Created: {goal_path}"
    )

    print("\nLeading observed opportunity:")

    print(
        f"  {leading['opportunity']}"
    )

    print(
        f"  18-month amount: "
        f"{money(leading_observed)}"
    )

    print(
        f"  Approx. quarterly amount: "
        f"{money(leading_quarterly)}"
    )

    print(
        f"  Illustrative 15% quarterly value: "
        f"{money(leading_15)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These figures identify the financial opportunity."
    )

    print(
        "They do not yet prove that the AI tool can deliver "
        "the illustrative 15% reduction."
    )

    print(
        "\nNext step: build the AI-assisted weekly support digest."
    )


if __name__ == "__main__":
    main()
