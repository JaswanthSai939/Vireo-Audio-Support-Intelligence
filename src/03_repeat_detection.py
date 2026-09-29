from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# VIREO AUDIO - STEP 3
# REPEAT-CONTACT DETECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

LOOKBACK_DAYS = 30

# These are intentionally conservative.
# We will validate the resulting candidates later.
SIMILARITY_THRESHOLD = 0.35

REPEAT_PHRASES = [
    "again",
    "same issue",
    "same problem",
    "previous ticket",
    "previous complaint",
    "earlier ticket",
    "earlier complaint",
    "already told",
    "already contacted",
    "contacted before",
    "wrote before",
    "writing again",
    "followed up",
    "follow up",
    "still not resolved",
    "not resolved",
    "no resolution",
    "nothing changed",
    "same as before",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_tickets():

    path = DATA_DIR / "tickets.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}"
        )

    return pd.read_csv(path)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Keep letters and numbers.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DEDUPLICATE
# ============================================================

def deduplicate_tickets(tickets):

    tickets = tickets.copy()

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce"
    )

    tickets["resolved_at"] = pd.to_datetime(
        tickets["resolved_at"],
        errors="coerce"
    )

    # Prefer current helpdesk records over legacy records.
    tickets["_source_priority"] = (
        tickets["source_system"]
        .astype(str)
        .str.lower()
        .eq("helpdesk")
        .astype(int)
    )

    # Prefer more complete records when source is equal.
    tickets["_completeness"] = (
        tickets.notna().sum(axis=1)
    )

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
        ]
    )

    tickets = tickets.drop_duplicates(
        subset="ticket_id",
        keep="first"
    ).copy()

    tickets.drop(
        columns=[
            "_source_priority",
            "_completeness",
        ],
        inplace=True
    )

    # Reset the index after deduplication.
    # TF-IDF matrix rows use a continuous 0...N-1 index.
    tickets.reset_index(
        drop=True,
        inplace=True
    )

    return tickets


# ============================================================
# REPEAT-PHRASE DETECTION
# ============================================================

def contains_repeat_language(text):

    text = clean_text(text)

    if not text:
        return False

    for phrase in REPEAT_PHRASES:

        if phrase in text:
            return True

    return False


# ============================================================
# MAIN REPEAT ANALYSIS
# ============================================================

def detect_repeat_candidates(tickets):

    df = tickets.copy()

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df["customer_message_clean"] = (
        df["customer_message"]
        .apply(clean_text)
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    df["resolved_at"] = pd.to_datetime(
        df["resolved_at"],
        errors="coerce"
    )

    df["repeat_language"] = (
        df["customer_message"]
        .apply(contains_repeat_language)
    )

    # --------------------------------------------------------
    # Prepare TF-IDF
    # --------------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000
    )

    text_matrix = vectorizer.fit_transform(
        df["customer_message_clean"]
    )

    # --------------------------------------------------------
    # Candidate matching
    # --------------------------------------------------------

    candidate_previous_ticket = [None] * len(df)
    candidate_similarity = np.zeros(
        len(df),
        dtype=float
    )

    candidate_time_gap = [None] * len(df)

    # Process customers independently.
    customer_groups = df.groupby(
        "customer_id",
        sort=False
    )

    print(
        f"Checking {len(customer_groups):,} customers..."
    )

    for customer_id, group in customer_groups:

        # Need at least two tickets.
        if len(group) < 2:
            continue

        group = group.sort_values(
            "created_at"
        )

        indices = group.index.tolist()

        for position in range(1, len(indices)):

            current_index = indices[position]

            current_date = df.loc[
                current_index,
                "created_at"
            ]

            if pd.isna(current_date):
                continue

            # Look backwards only within 30 days.
            previous_indices = []

            for previous_position in range(
                position - 1,
                -1,
                -1
            ):

                previous_index = indices[
                    previous_position
                ]

                previous_date = df.loc[
                    previous_index,
                    "created_at"
                ]

                if pd.isna(previous_date):
                    continue

                gap_days = (
                    current_date - previous_date
                ).total_seconds() / 86400

                if gap_days < 0:
                    continue

                if gap_days > LOOKBACK_DAYS:
                    break

                previous_indices.append(
                    previous_index
                )

            if not previous_indices:
                continue

            # ------------------------------------------------
            # Compare current ticket against prior tickets.
            # ------------------------------------------------

            current_vector = text_matrix[
                current_index
            ]

            best_index = None
            best_similarity = 0.0
            best_gap = None

            for previous_index in previous_indices:

                previous_vector = text_matrix[
                    previous_index
                ]

                similarity = cosine_similarity(
                    current_vector,
                    previous_vector
                )[0][0]

                previous_date = df.loc[
                    previous_index,
                    "created_at"
                ]

                gap_days = (
                    current_date - previous_date
                ).total_seconds() / 86400

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_index = previous_index
                    best_gap = gap_days

            if best_index is not None:

                candidate_previous_ticket[
                    current_index
                ] = df.loc[
                    best_index,
                    "ticket_id"
                ]

                candidate_similarity[
                    current_index
                ] = best_similarity

                candidate_time_gap[
                    current_index
                ] = best_gap

    # --------------------------------------------------------
    # Add candidate signals
    # --------------------------------------------------------

    df["previous_ticket_id"] = (
        candidate_previous_ticket
    )

    df["previous_similarity"] = (
        candidate_similarity
    )

    df["days_since_previous_contact"] = (
        candidate_time_gap
    )

    # --------------------------------------------------------
    # Structured same-product/category signal
    # --------------------------------------------------------

    df["same_product_as_previous"] = False
    df["same_category_as_previous"] = False

    valid_previous = (
        df["previous_ticket_id"].notna()
    )

    ticket_lookup = df.set_index(
        "ticket_id"
    )

    for index in df.index[valid_previous]:

        previous_ticket_id = df.loc[
            index,
            "previous_ticket_id"
        ]

        if previous_ticket_id not in ticket_lookup.index:
            continue

        previous_row = ticket_lookup.loc[
            previous_ticket_id
        ]

        current_product = df.loc[
            index,
            "product_sku"
        ]

        previous_product = previous_row[
            "product_sku"
        ]

        current_category = df.loc[
            index,
            "category"
        ]

        previous_category = previous_row[
            "category"
        ]

        df.loc[
            index,
            "same_product_as_previous"
        ] = (
            current_product == previous_product
        )

        df.loc[
            index,
            "same_category_as_previous"
        ] = (
            current_category == previous_category
        )

    # --------------------------------------------------------
    # Candidate classification
    # --------------------------------------------------------

    df["repeat_candidate"] = (
        (
            df["previous_similarity"]
            >= SIMILARITY_THRESHOLD
        )
        & (
            df["days_since_previous_contact"]
            <= LOOKBACK_DAYS
        )
    )

    # Stronger signal:
    # Similar text + same product/category OR explicit
    # repeat language.
    df["strong_repeat_candidate"] = (
        df["repeat_candidate"]
        & (
            df["same_product_as_previous"]
            | df["same_category_as_previous"]
            | df["repeat_language"]
        )
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def build_summary(df):

    total_tickets = len(df)

    candidate_count = int(
        df["repeat_candidate"].sum()
    )

    strong_candidate_count = int(
        df["strong_repeat_candidate"].sum()
    )

    candidate_rate = (
        candidate_count
        / total_tickets
        * 100
    )

    strong_candidate_rate = (
        strong_candidate_count
        / total_tickets
        * 100
    )

    # CSAT comparison.
    csat = pd.to_numeric(
        df["csat_score"],
        errors="coerce"
    )

    # Policy: zero means no response.
    csat = csat.mask(csat == 0)

    candidate_csat = csat[
        df["strong_repeat_candidate"]
    ].mean()

    non_candidate_csat = csat[
        ~df["strong_repeat_candidate"]
    ].mean()

    # Transfer comparison.
    transfers = pd.to_numeric(
        df["transfers"],
        errors="coerce"
    ).fillna(0)

    candidate_transfer_rate = (
        transfers[
            df["strong_repeat_candidate"]
        ] > 0
    ).mean() * 100

    non_candidate_transfer_rate = (
        transfers[
            ~df["strong_repeat_candidate"]
        ] > 0
    ).mean() * 100

    return {
        "total_unique_tickets": total_tickets,

        "repeat_candidates": candidate_count,

        "repeat_candidate_rate_percent": (
            candidate_rate
        ),

        "strong_repeat_candidates": (
            strong_candidate_count
        ),

        "strong_repeat_candidate_rate_percent": (
            strong_candidate_rate
        ),

        "average_csat_repeat_candidates": (
            float(candidate_csat)
            if pd.notna(candidate_csat)
            else None
        ),

        "average_csat_non_repeat": (
            float(non_candidate_csat)
            if pd.notna(non_candidate_csat)
            else None
        ),

        "transfer_rate_repeat_candidates_percent": (
            float(candidate_transfer_rate)
        ),

        "transfer_rate_non_repeat_percent": (
            float(non_candidate_transfer_rate)
        ),
    }


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(df, summary):

    # --------------------------------------------------------
    # Save detailed candidate file
    # --------------------------------------------------------

    candidate_columns = [
        "ticket_id",
        "customer_id",
        "created_at",
        "resolved_at",
        "product_sku",
        "category",
        "channel",
        "priority",
        "customer_message",
        "previous_ticket_id",
        "days_since_previous_contact",
        "previous_similarity",
        "same_product_as_previous",
        "same_category_as_previous",
        "repeat_language",
        "repeat_candidate",
        "strong_repeat_candidate",
        "csat_score",
        "transfers",
    ]

    candidate_columns = [
        column
        for column in candidate_columns
        if column in df.columns
    ]

    candidate_df = df[
        candidate_columns
    ].copy()

    candidate_df = candidate_df[
        candidate_df["strong_repeat_candidate"]
    ].sort_values(
        [
            "previous_similarity",
            "days_since_previous_contact"
        ],
        ascending=[
            False,
            True
        ]
    )

    candidate_path = (
        REPORTS_DIR /
        "repeat_candidates.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        REPORTS_DIR /
        "repeat_analysis.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save Markdown
    # --------------------------------------------------------

    md_path = (
        REPORTS_DIR /
        "03_repeat_analysis.md"
    )

    md_content = f"""# Vireo Audio — Repeat Contact Analysis

## Purpose

This analysis identifies **potential repeat contacts** where the
same customer contacts support again within a 30-day window.

The detector is a candidate-generation tool, not ground truth.
Its accuracy must be manually validated before the repeat-contact
rate is used as a final business KPI.

---

## Population

- Unique tickets analyzed: **{summary["total_unique_tickets"]:,}**

---

## Candidate Results

| Metric | Result |
|---|---:|
| Repeat candidates | {summary["repeat_candidates"]:,} |
| Repeat candidate rate | {summary["repeat_candidate_rate_percent"]:.2f}% |
| Strong repeat candidates | {summary["strong_repeat_candidates"]:,} |
| Strong candidate rate | {summary["strong_repeat_candidate_rate_percent"]:.2f}% |

---

## Detection Signals

A candidate is identified using:

1. Same customer.
2. Previous contact within 30 days.
3. TF-IDF cosine similarity between customer messages.
4. Additional supporting signals:
   - Same product.
   - Same category.
   - Explicit repeat-contact language.

The similarity threshold used in this analysis is:

**{SIMILARITY_THRESHOLD:.2f}**

---

## Customer Experience Comparison

| Metric | Repeat Candidates | Other Tickets |
|---|---:|---:|
| Average CSAT | {
        f"{summary['average_csat_repeat_candidates']:.2f}"
        if summary["average_csat_repeat_candidates"] is not None
        else "N/A"
    } | {
        f"{summary['average_csat_non_repeat']:.2f}"
        if summary["average_csat_non_repeat"] is not None
        else "N/A"
    } |
| Tickets with transfer | {
        f"{summary['transfer_rate_repeat_candidates_percent']:.2f}%"
    } | {
        f"{summary['transfer_rate_non_repeat_percent']:.2f}%"
    } |

---

## Important Limitation

The detector can produce false positives when the same customer
contacts support about a different issue involving the same or
similar product.

It can also miss repeat contacts when customers describe the same
problem using substantially different language.

Therefore, the candidate rate should **not** yet be presented as
a confirmed repeat-contact rate.

A manually reviewed validation sample is required before selecting
the final business target.

---

## Output

Detailed strong-repeat candidates are stored in:

`reports/repeat_candidates.csv`
"""

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(md_content)

    return (
        candidate_path,
        json_path,
        md_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VIREO AUDIO - STEP 3: REPEAT CONTACT DETECTION")
    print("=" * 70)

    print("\n[1/4] Loading tickets...")

    tickets = load_tickets()

    print(
        f"  Raw rows: {len(tickets):,}"
    )

    print("\n[2/4] Deduplicating tickets...")

    tickets = deduplicate_tickets(
        tickets
    )

    print(
        f"  Unique tickets: {len(tickets):,}"
    )

    print("\n[3/4] Detecting repeat-contact candidates...")

    result = detect_repeat_candidates(
        tickets
    )

    print("\nBuilding summary...")

    summary = build_summary(
        result
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REPEAT-CONTACT RESULTS")
    print("=" * 70)

    print(
        f"\nUnique tickets: "
        f"{summary['total_unique_tickets']:,}"
    )

    print(
        f"Repeat candidates: "
        f"{summary['repeat_candidates']:,}"
    )

    print(
        f"Repeat candidate rate: "
        f"{summary['repeat_candidate_rate_percent']:.2f}%"
    )

    print(
        f"Strong repeat candidates: "
        f"{summary['strong_repeat_candidates']:,}"
    )

    print(
        f"Strong candidate rate: "
        f"{summary['strong_repeat_candidate_rate_percent']:.2f}%"
    )

    print(
        f"\nAverage CSAT — candidates: "
        f"{summary['average_csat_repeat_candidates']:.2f}"
        if summary["average_csat_repeat_candidates"]
        is not None
        else "\nAverage CSAT — candidates: N/A"
    )

    print(
        f"Average CSAT — other tickets: "
        f"{summary['average_csat_non_repeat']:.2f}"
        if summary["average_csat_non_repeat"]
        is not None
        else "Average CSAT — other tickets: N/A"
    )

    print(
        f"\nTransfer rate — candidates: "
        f"{summary['transfer_rate_repeat_candidates_percent']:.2f}%"
    )

    print(
        f"Transfer rate — other tickets: "
        f"{summary['transfer_rate_non_repeat_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[4/4] Saving reports...")

    (
        candidate_path,
        json_path,
        md_path
    ) = save_reports(
        result,
        summary
    )

    print(
        f"\n✓ Created: {candidate_path}"
    )

    print(
        f"✓ Created: {json_path}"
    )

    print(
        f"✓ Created: {md_path}"
    )

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT: Candidate rate is not yet a validated KPI."
    )

    print(
        "Next step: manually validate a sample of candidates."
    )


if __name__ == "__main__":
    main()
