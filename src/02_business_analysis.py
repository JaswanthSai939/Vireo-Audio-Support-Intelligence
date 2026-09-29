from pathlib import Path

import json

import pandas as pd

# ============================================================

# VIREO AUDIO - STEP 2

# BUSINESS ANALYSIS

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================

# POLICY COSTS

# ============================================================

CHANNEL_COSTS = {

    "chat": 210,

    "email": 260,

    "voice": 520,

    "social": 240,

}

BLENDED_CONTACT_COST = 290

SLA_BREACH_CREDIT = 350

TRANSFER_COST = 305

REPLACEMENT_LOGISTICS_COST = 340

SLA_TARGET_MINUTES = {

    "chat": 15,

    "voice": 120,

    "social": 240,

    "email": 480,

}

# ============================================================

# LOAD DATA

# ============================================================


def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():

        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)

# ============================================================

# DEDUPLICATE TICKETS

# ============================================================


def deduplicate_tickets(tickets):
    """

    Produce one analytical record per ticket_id.

    The supplied context says some legacy tickets were re-imported

    into the current helpdesk.

    Decision:

    1. Prefer current helpdesk records over legacy_fd records.

    2. Within the same source system, prefer the row with the

       greatest number of populated fields.

    """

    tickets = tickets.copy()

    tickets["_source_priority"] = (

        tickets["source_system"]

        .astype(str)

        .str.lower()

        .eq("helpdesk")

        .astype(int)

    )

    tickets["_completeness"] = tickets.notna().sum(axis=1)

    tickets = tickets.sort_values(

        by=[

            "ticket_id",

            "_source_priority",

            "_completeness",

        ],

        ascending=[

            True,

            False,

            False,

        ],

    )

    duplicate_count = int(

        tickets["ticket_id"].duplicated(keep=False).sum()

    )

    duplicate_ids = int(

        tickets.loc[

            tickets["ticket_id"].duplicated(keep=False),

            "ticket_id"

        ].nunique()

    )

    clean = tickets.drop_duplicates(

        subset="ticket_id",

        keep="first"

    ).copy()

    clean = clean.drop(

        columns=[

            "_source_priority",

            "_completeness",

        ]

    )

    return clean, duplicate_ids, duplicate_count

# ============================================================

# SLA ANALYSIS

# ============================================================


def calculate_sla_metrics(tickets):

    df = tickets.copy()

    df["created_at"] = pd.to_datetime(

        df["created_at"],

        errors="coerce"

    )

    df["first_response_at"] = pd.to_datetime(

        df["first_response_at"],

        errors="coerce"

    )

    df["response_minutes"] = (

        df["first_response_at"] - df["created_at"]

    ).dt.total_seconds() / 60

    df["sla_target_minutes"] = (

        df["channel"]

        .map(SLA_TARGET_MINUTES)

    )

    # A breach requires a valid human response timestamp.

    df["sla_breach"] = (

        df["response_minutes"].notna()

        & df["sla_target_minutes"].notna()

        & (

            df["response_minutes"]

            > df["sla_target_minutes"]

        )

    )

    breach_count = int(df["sla_breach"].sum())

    breach_cost = (

        breach_count * SLA_BREACH_CREDIT

    )

    return df, breach_count, breach_cost

# ============================================================

# CONTACT COST

# ============================================================


def calculate_contact_cost(tickets):

    channel_counts = (

        tickets["channel"]

        .value_counts()

        .to_dict()

    )

    channel_cost = 0

    for channel, count in channel_counts.items():

        unit_cost = CHANNEL_COSTS.get(

            str(channel).lower(),

            BLENDED_CONTACT_COST

        )

        channel_cost += count * unit_cost

    blended_cost = (

        len(tickets) * BLENDED_CONTACT_COST

    )

    return channel_counts, channel_cost, blended_cost

# ============================================================

# TRANSFER ANALYSIS

# ============================================================


def calculate_transfer_metrics(tickets):

    transfers = pd.to_numeric(

        tickets["transfers"],

        errors="coerce"

    ).fillna(0)

    total_transfers = int(

        transfers.sum()

    )

    transfer_cost = (

        total_transfers * TRANSFER_COST

    )

    tickets_with_transfers = int(

        (transfers > 0).sum()

    )

    return (

        total_transfers,

        tickets_with_transfers,

        transfer_cost,

    )

# ============================================================

# REFUND ANALYSIS

# ============================================================


def calculate_refund_metrics(tickets):

    refunds = pd.to_numeric(

        tickets["refund_amount_inr"],

        errors="coerce"

    ).fillna(0)

    refund_count = int(

        (refunds > 0).sum()

    )

    refund_total = float(

        refunds.sum()

    )

    refund_reasons = (

        tickets.loc[

            refunds > 0,

            "refund_reason_code"

        ]

        .fillna("UNKNOWN")

        .value_counts()

        .to_dict()

    )

    return (

        refund_count,

        refund_total,

        refund_reasons,

    )

# ============================================================

# REPLACEMENT ANALYSIS

# ============================================================


def calculate_replacement_metrics(

    tickets,

    products

):
    """

    Calculate replacement count and estimated replacement cost.

    Policy:

        Replacement cost =

        product unit cost + ₹340 reverse/forward shipping.

    """

    # Work on copies so the original DataFrames are not modified.

    ticket_data = tickets.copy()

    product_data = products.copy()

    # Normalize replacement flag.

    ticket_data["replacement_flag"] = (

        ticket_data["replacement_issued"]

        .astype("string")

        .str.strip()

        .str.upper()

        .eq("Y")

    )

    # Make sure product unit cost is numeric.

    product_data["unit_cost_inr"] = pd.to_numeric(

        product_data["unit_cost_inr"],

        errors="coerce"

    )

    # Keep only the columns needed for the join.

    product_costs = product_data[

        ["sku", "unit_cost_inr"]

    ].drop_duplicates(

        subset=["sku"]

    )

    # Join product cost onto each ticket.

    merged = ticket_data.merge(

        product_costs,

        left_on="product_sku",

        right_on="sku",

        how="left"

    )

    # Count replacements after the merge.

    replacement_mask = merged["replacement_flag"]

    replacement_count = int(

        replacement_mask.sum()

    )

    # Check how many replacement tickets don't have

    # a usable product unit cost.

    missing_unit_cost = int(

        merged.loc[

            replacement_mask,

            "unit_cost_inr"

        ].isna().sum()

    )

    # Calculate replacement cost.

    merged["replacement_cost"] = 0.0

    valid_replacement_mask = (

        replacement_mask

        & merged["unit_cost_inr"].notna()

    )

    merged.loc[

        valid_replacement_mask,

        "replacement_cost"

    ] = (

        merged.loc[

            valid_replacement_mask,

            "unit_cost_inr"

        ]

        + REPLACEMENT_LOGISTICS_COST

    )

    replacement_cost = float(

        merged["replacement_cost"].sum()

    )

    return (

        replacement_count,

        replacement_cost,

        missing_unit_cost,

    )

# ============================================================

# CSAT ANALYSIS

# ============================================================


def calculate_csat_metrics(tickets):

    csat = pd.to_numeric(

        tickets["csat_score"],

        errors="coerce"

    )

    # The supplied policy says legacy 0 means no response.

    csat = csat.mask(csat == 0)

    valid = csat[

        csat.between(1, 5)

    ]

    response_count = int(

        valid.notna().sum()

    )

    average_csat = (

        float(valid.mean())

        if len(valid) > 0

        else None

    )

    low_csat_count = int(

        (valid <= 2).sum()

    )

    high_csat_count = int(

        (valid >= 4).sum()

    )

    return (

        response_count,

        average_csat,

        low_csat_count,

        high_csat_count,

    )

# ============================================================

# CATEGORY ANALYSIS

# ============================================================


def calculate_category_analysis(tickets):

    categories = (

        tickets["category"]

        .fillna("Unknown")

        .astype(str)

        .str.strip()

    )

    counts = (

        categories

        .value_counts()

        .reset_index()

    )

    counts.columns = [

        "category",

        "ticket_count"

    ]

    counts["percentage"] = (

        counts["ticket_count"]

        / len(tickets)

        * 100

    )

    return counts

# ============================================================

# CHANNEL ANALYSIS

# ============================================================


def calculate_channel_analysis(tickets):

    result = (

        tickets.groupby("channel")

        .agg(

            tickets=("ticket_id", "count"),

            avg_transfers=("transfers", "mean"),

        )

        .reset_index()

    )

    result["contact_cost"] = (

        result["channel"]

        .map(CHANNEL_COSTS)

        * result["tickets"]

    )

    return result

# ============================================================

# TEAM ANALYSIS

# ============================================================


def calculate_team_analysis(tickets):

    result = (

        tickets.groupby("assigned_team")

        .agg(

            tickets=("ticket_id", "count"),

            avg_transfers=("transfers", "mean"),

            avg_csat=("csat_score", "mean"),

        )

        .reset_index()

        .sort_values(

            "tickets",

            ascending=False

        )

    )

    return result

# ============================================================

# WEEKLY ANALYSIS

# ============================================================


def calculate_weekly_analysis(tickets):

    df = tickets.copy()

    df["created_at"] = pd.to_datetime(

        df["created_at"],

        errors="coerce"

    )

    df["week"] = (

        df["created_at"]

        .dt.to_period("W")

        .astype(str)

    )

    weekly = (

        df.groupby("week")

        .agg(

            tickets=("ticket_id", "count"),

            avg_transfers=("transfers", "mean"),

        )

        .reset_index()

    )

    return weekly

# ============================================================

# PRIORITY ANALYSIS

# ============================================================


def calculate_priority_analysis(tickets):

    result = (

        tickets.groupby("priority")

        .agg(

            tickets=("ticket_id", "count"),

            avg_transfers=("transfers", "mean"),

        )

        .reset_index()

    )

    return result

# ============================================================

# BUSINESS OPPORTUNITIES

# ============================================================


def build_business_opportunities(

    tickets,

    sla_breach_count,

    sla_breach_cost,

    transfer_cost,

    refund_total,

    replacement_cost,

):

    rows = []

    # SLA

    rows.append({

        "opportunity": "SLA breach credits",

        "metric": "First-response breaches",

        "count": sla_breach_count,

        "estimated_cost_inr": sla_breach_cost,

        "basis": "₹350 automatic credit per breach",

    })

    # Transfers

    transfers = pd.to_numeric(

        tickets["transfers"],

        errors="coerce"

    ).fillna(0)

    total_transfers = int(

        transfers.sum()

    )

    rows.append({

        "opportunity": "Internal transfers",

        "metric": "Team hand-offs",

        "count": total_transfers,

        "estimated_cost_inr": transfer_cost,

        "basis": "₹305 per internal transfer",

    })

    # Refunds

    refund_count = int(

        (

            pd.to_numeric(

                tickets["refund_amount_inr"],

                errors="coerce"

            ).fillna(0)

            > 0

        ).sum()

    )

    rows.append({

        "opportunity": "Refunds",

        "metric": "Refunded tickets",

        "count": refund_count,

        "estimated_cost_inr": refund_total,

        "basis": "Exported refund amount",

    })

    # Replacements

    replacement_count = int(

        tickets["replacement_issued"]

        .astype(str)

        .str.upper()

        .eq("Y")

        .sum()

    )

    rows.append({

        "opportunity": "Replacements",

        "metric": "Replacement tickets",

        "count": replacement_count,

        "estimated_cost_inr": replacement_cost,

        "basis": "Product unit cost + ₹340 logistics",

    })

    result = pd.DataFrame(rows)

    return result.sort_values(

        "estimated_cost_inr",

        ascending=False

    )

# ============================================================

# MAIN

# ============================================================


def main():

    print("=" * 70)

    print("VIREO AUDIO - STEP 2: BUSINESS ANALYSIS")

    print("=" * 70)

    # --------------------------------------------------------

    # Load

    # --------------------------------------------------------

    print("\n[1/9] Loading datasets...")

    tickets_raw = load_csv("tickets.csv")

    products = load_csv("products.csv")

    print(

        f"  Raw tickets: "

        f"{len(tickets_raw):,}"

    )

    # --------------------------------------------------------

    # Deduplicate

    # --------------------------------------------------------

    print("\n[2/9] Deduplicating ticket records...")

    tickets, duplicate_ids, duplicate_rows = (

        deduplicate_tickets(tickets_raw)

    )

    print(

        f"  Duplicate ticket IDs found: "

        f"{duplicate_ids:,}"

    )

    print(

        f"  Rows involved in duplicates: "

        f"{duplicate_rows:,}"

    )

    print(

        f"  Analytical ticket population: "

        f"{len(tickets):,}"

    )

    # --------------------------------------------------------

    # Contact cost

    # --------------------------------------------------------

    print("\n[3/9] Calculating contact costs...")

    (

        channel_counts,

        channel_cost,

        blended_cost

    ) = calculate_contact_cost(tickets)

    print(

        f"  Channel-specific estimated cost: "

        f"₹{channel_cost:,.2f}"

    )

    print(

        f"  Blended ₹290 estimate: "

        f"₹{blended_cost:,.2f}"

    )

    # --------------------------------------------------------

    # SLA

    # --------------------------------------------------------

    print("\n[4/9] Calculating SLA breaches...")

    (

        tickets,

        sla_breach_count,

        sla_breach_cost

    ) = calculate_sla_metrics(tickets)

    print(

        f"  SLA breaches: "

        f"{sla_breach_count:,}"

    )

    print(

        f"  SLA credit exposure: "

        f"₹{sla_breach_cost:,.2f}"

    )

    # --------------------------------------------------------

    # Transfers

    # --------------------------------------------------------

    print("\n[5/9] Calculating transfer costs...")

    (

        total_transfers,

        tickets_with_transfers,

        transfer_cost

    ) = calculate_transfer_metrics(tickets)

    print(

        f"  Total transfers: "

        f"{total_transfers:,}"

    )

    print(

        f"  Tickets with transfers: "

        f"{tickets_with_transfers:,}"

    )

    print(

        f"  Transfer cost: "

        f"₹{transfer_cost:,.2f}"

    )

    # --------------------------------------------------------

    # Refunds

    # --------------------------------------------------------

    print("\n[6/9] Calculating refunds and replacements...")

    (

        refund_count,

        refund_total,

        refund_reasons

    ) = calculate_refund_metrics(tickets)

    print(

        f"  Refund tickets: "

        f"{refund_count:,}"

    )

    print(

        f"  Refund amount: "

        f"₹{refund_total:,.2f}"

    )

    (

        replacement_count,

        replacement_cost,

        missing_unit_cost

    ) = calculate_replacement_metrics(

        tickets,

        products

    )

    print(

        f"  Replacement tickets: "

        f"{replacement_count:,}"

    )

    print(

        f"  Estimated replacement cost: "

        f"₹{replacement_cost:,.2f}"

    )

    if missing_unit_cost > 0:

        print(

            f"  WARNING: {missing_unit_cost:,} "

            f"replacement rows missing product cost"

        )

    # --------------------------------------------------------

    # CSAT

    # --------------------------------------------------------

    print("\n[7/9] Calculating CSAT...")

    (

        csat_responses,

        average_csat,

        low_csat_count,

        high_csat_count

    ) = calculate_csat_metrics(tickets)

    print(

        f"  Valid CSAT responses: "

        f"{csat_responses:,}"

    )

    print(

        f"  Average CSAT: "

        f"{average_csat:.2f}"

        if average_csat is not None

        else "  Average CSAT: N/A"

    )

    print(

        f"  Low CSAT (1-2): "

        f"{low_csat_count:,}"

    )

    print(

        f"  High CSAT (4-5): "

        f"{high_csat_count:,}"

    )

    # --------------------------------------------------------

    # Categories

    # --------------------------------------------------------

    print("\n[8/9] Analyzing complaint categories...")

    category_analysis = calculate_category_analysis(

        tickets

    )

    print("\n  Top 10 categories:")

    for _, row in category_analysis.head(10).iterrows():

        print(

            f"    {row['category']:<25}"

            f"{int(row['ticket_count']):>6} tickets "

            f"({row['percentage']:.2f}%)"

        )

    # --------------------------------------------------------

    # Additional analyses

    # --------------------------------------------------------

    print("\n[9/9] Building business analysis tables...")

    channel_analysis = calculate_channel_analysis(

        tickets

    )

    team_analysis = calculate_team_analysis(

        tickets

    )

    weekly_analysis = calculate_weekly_analysis(

        tickets

    )

    priority_analysis = calculate_priority_analysis(

        tickets

    )

    opportunities = build_business_opportunities(

        tickets=tickets,

        sla_breach_count=sla_breach_count,

        sla_breach_cost=sla_breach_cost,

        transfer_cost=transfer_cost,

        refund_total=refund_total,

        replacement_cost=replacement_cost,

    )

    # ========================================================

    # SAVE OPPORTUNITIES CSV

    # ========================================================

    opportunities_path = (

        REPORTS_DIR /

        "business_opportunities.csv"

    )

    opportunities.to_csv(

        opportunities_path,

        index=False

    )

    # ========================================================

    # BUILD JSON

    # ========================================================

    analysis_report = {

        "ticket_population": {

            "raw_rows": int(len(tickets_raw)),

            "unique_ticket_population": int(len(tickets)),

            "duplicate_ticket_ids": int(duplicate_ids),

            "duplicate_rows": int(duplicate_rows),

        },

        "contact_cost": {

            "channel_specific_inr": float(channel_cost),

            "blended_290_inr": float(blended_cost),

            "channel_counts": {

                str(k): int(v)

                for k, v in channel_counts.items()

            },

        },

        "sla": {

            "breach_count": int(sla_breach_count),

            "breach_rate_percent": (

                sla_breach_count

                / len(tickets)

                * 100

            ),

            "credit_exposure_inr": float(

                sla_breach_cost

            ),

        },

        "transfers": {

            "total_transfers": int(

                total_transfers

            ),

            "tickets_with_transfers": int(

                tickets_with_transfers

            ),

            "transfer_cost_inr": float(

                transfer_cost

            ),

        },

        "refunds": {

            "refund_ticket_count": int(

                refund_count

            ),

            "refund_total_inr": float(

                refund_total

            ),

            "refund_reasons": {

                str(k): int(v)

                for k, v in refund_reasons.items()

            },

        },

        "replacements": {

            "replacement_ticket_count": int(

                replacement_count

            ),

            "estimated_cost_inr": float(

                replacement_cost

            ),

            "missing_product_cost_rows": int(

                missing_unit_cost

            ),

        },

        "csat": {

            "valid_responses": int(

                csat_responses

            ),

            "average": (

                float(average_csat)

                if average_csat is not None

                else None

            ),

            "low_1_to_2": int(

                low_csat_count

            ),

            "high_4_to_5": int(

                high_csat_count

            ),

        },

        "top_categories": (

            category_analysis

            .head(10)

            .to_dict(orient="records")

        ),

        "business_opportunities": (

            opportunities

            .to_dict(orient="records")

        ),

    }

    json_path = (

        REPORTS_DIR /

        "business_analysis.json"

    )

    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            analysis_report,

            file,

            indent=4,

            ensure_ascii=False

        )

    # ========================================================

    # BUILD MARKDOWN REPORT

    # ========================================================

    md_path = (

        REPORTS_DIR /

        "02_business_analysis.md"

    )

    top_categories_text = ""

    for _, row in category_analysis.head(10).iterrows():

        top_categories_text += (

            f"| {row['category']} | "

            f"{int(row['ticket_count']):,} | "

            f"{row['percentage']:.2f}% |\n"

        )

    opportunities_text = ""

    for _, row in opportunities.iterrows():

        opportunities_text += (

            f"| {row['opportunity']} | "

            f"{row['metric']} | "

            f"{int(row['count']):,} | "

            f"₹{row['estimated_cost_inr']:,.2f} | "

            f"{row['basis']} |\n"

        )

    md_content = f"""# Vireo Audio — Business Analysis

\## 1. Analytical Population

\- Raw ticket rows: {len(tickets_raw):,}

\- Unique ticket population after deduplication: {len(tickets):,}

\- Duplicate ticket IDs: {duplicate_ids:,}

The duplicate handling decision is documented in the Step 1 audit.

\---

\## 2. Contact Cost

\| Measure | Estimated Cost |

\|---|---:|

\| Channel-specific cost | ₹{channel_cost:,.2f} |

\| Blended ₹290/contact | ₹{blended_cost:,.2f} |

The channel-specific estimate uses the policy's channel costs.

The blended estimate uses Vireo's stated ₹290 blended cost per contact.

\---

\## 3. SLA Breaches

\- SLA breaches: {sla_breach_count:,}

\- Breach rate: {sla_breach_count / len(tickets) * 100:.2f}%

\- Automatic credit exposure: ₹{sla_breach_cost:,.2f}

The policy specifies ₹350 of store credit for every first-response SLA breach.

\---

\## 4. Internal Transfers

\- Total transfers: {total_transfers:,}

\- Tickets with at least one transfer: {tickets_with_transfers:,}

\- Estimated transfer cost: ₹{transfer_cost:,.2f}

The policy assigns ₹305 of cost to each internal transfer.

\---

\## 5. Refunds

\- Refund tickets: {refund_count:,}

\- Exported refund amount: ₹{refund_total:,.2f}

\### Refund reasons

"""

    for reason, count in refund_reasons.items():

        md_content += (

            f"- {reason}: {count:,} tickets\n"

        )

    md_content += f"""

\---

\## 6. Replacements

\- Replacement tickets: {replacement_count:,}

\- Estimated replacement cost: ₹{replacement_cost:,.2f}

\- Replacement rows missing product cost: {missing_unit_cost:,}

Replacement cost is calculated as:

Product unit cost + ₹340 reverse/forward shipping

\---

\## 7. CSAT

\- Valid CSAT responses: {csat_responses:,}

\- Average CSAT: {

        f"{average_csat:.2f}"

        if average_csat is not None

        else "N/A"

    }

\- Low CSAT (1–2): {low_csat_count:,}

\- High CSAT (4–5): {high_csat_count:,}

CSAT values of zero are treated as no response for legacy records.

\---

\## 8. Top Complaint Categories

\| Category | Tickets | Share |

\|---|---:|---:|

{top_categories_text}

\---

\## 9. Business Opportunity Candidates

\| Opportunity | Metric | Count | Estimated Cost | Basis |

\|---|---|---:|---:|---|

{opportunities_text}

\---

\## 10. Important Interpretation

These figures identify business-impact candidates.

They do not by themselves prove that one category or process caused the associated cost.

The next analysis stage will investigate the ticket text and identify recurring customer issues, including potential repeat contacts.

The final business goal will be selected only after validating the relevant pattern in the underlying ticket data.

"""

    with open(

        md_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(md_content)

    # ========================================================

    # FINAL OUTPUT

    # ========================================================

    print("\n" + "=" * 70)

    print("STEP 2 BUSINESS ANALYSIS COMPLETE")

    print("=" * 70)

    print("\nReports created:")

    print(

        f"✓ {json_path}"

    )

    print(

        f"✓ {md_path}"

    )

    print(

        f"✓ {opportunities_path}"

    )

    print("\nTop business opportunity candidates:")

    for _, row in opportunities.iterrows():

        print(

            f"  {row['opportunity']:<25}"

            f"₹{row['estimated_cost_inr']:,.2f}"

        )

    print("\nNext step:")

    print(

        "Use these results to investigate the strongest "

        "business problem before selecting the final goal."

    )


if __name__ == "__main__":

    main()
