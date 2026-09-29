from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# VIREO AUDIO - STEP 4
# REPEAT DETECTOR VALIDATION SAMPLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 50
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 70)
    print("VIREO AUDIO - STEP 4: REPEAT DETECTOR VALIDATION")
    print("=" * 70)

    tickets_path = DATA_DIR / "tickets.csv"
    candidates_path = REPORTS_DIR / "repeat_candidates.csv"

    tickets = pd.read_csv(tickets_path)
    candidates = pd.read_csv(candidates_path)

    print(
        f"\nRaw tickets: {len(tickets):,}"
    )

    print(
        f"Strong repeat candidates: {len(candidates):,}"
    )

    # ========================================================
    # PREPARE TICKET LOOKUP
    # ========================================================

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce"
    )

    # Prefer helpdesk records when duplicates exist.
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

    # ========================================================
    # PREVIOUS TICKET INFORMATION
    # ========================================================

    previous_lookup = tickets[
        [
            "ticket_id",
            "created_at",
            "customer_message",
            "category",
            "product_sku",
            "channel"
        ]
    ].copy()

    previous_lookup = previous_lookup.rename(
        columns={
            "ticket_id": "previous_ticket_id",
            "created_at": "previous_created_at",
            "customer_message": "previous_customer_message",
            "category": "previous_category",
            "product_sku": "previous_product_sku",
            "channel": "previous_channel"
        }
    )

    # ========================================================
    # JOIN PREVIOUS TICKET
    # ========================================================

    candidates = candidates.merge(
        previous_lookup,
        on="previous_ticket_id",
        how="left"
    )

    # ========================================================
    # STRATIFIED SAMPLE
    # ========================================================

    # We don't want all 50 examples to be the easiest/highest
    # similarity cases. Divide candidates into similarity bands.

    candidates["similarity_band"] = pd.cut(
        candidates["previous_similarity"],
        bins=[
            -np.inf,
            0.45,
            0.60,
            0.75,
            np.inf
        ],
        labels=[
            "low",
            "medium",
            "high",
            "very_high"
        ]
    )

    bands = (
        candidates["similarity_band"]
        .value_counts()
        .sort_index()
    )

    print("\nSimilarity bands:")

    for band, count in bands.items():

        print(
            f"  {band}: {count:,}"
        )

    # Approximately equal sample from each band.
    available_bands = [
        band
        for band in candidates["similarity_band"].dropna().unique()
    ]

    samples = []

    per_band = max(
        1,
        SAMPLE_SIZE // max(
            len(available_bands),
            1
        )
    )

    for band in available_bands:

        band_data = candidates[
            candidates["similarity_band"] == band
        ]

        n = min(
            per_band,
            len(band_data)
        )

        samples.append(
            band_data.sample(
                n=n,
                random_state=RANDOM_SEED
            )
        )

    sample = pd.concat(
        samples,
        ignore_index=True
    )

    # Fill remaining slots if necessary.
    if len(sample) < SAMPLE_SIZE:

        remaining = candidates[
            ~candidates["ticket_id"].isin(
                sample["ticket_id"]
            )
        ]

        remaining_count = (
            SAMPLE_SIZE - len(sample)
        )

        if len(remaining) > 0:

            extra = remaining.sample(
                n=min(
                    remaining_count,
                    len(remaining)
                ),
                random_state=RANDOM_SEED
            )

            sample = pd.concat(
                [
                    sample,
                    extra
                ],
                ignore_index=True
            )

    sample = sample.head(
        SAMPLE_SIZE
    )

    # ========================================================
    # REVIEW COLUMNS
    # ========================================================

    sample["review_label"] = ""

    sample["review_notes"] = ""

    sample["reviewer"] = ""

    sample["review_date"] = ""

    review_columns = [
        "ticket_id",
        "customer_id",
        "created_at",
        "previous_ticket_id",
        "previous_created_at",
        "days_since_previous_contact",
        "previous_similarity",
        "similarity_band",
        "product_sku",
        "previous_product_sku",
        "category",
        "previous_category",
        "channel",
        "previous_channel",
        "customer_message",
        "previous_customer_message",
        "repeat_language",
        "review_label",
        "review_notes",
        "reviewer",
        "review_date"
    ]

    review_columns = [
        column
        for column in review_columns
        if column in sample.columns
    ]

    sample = sample[
        review_columns
    ]

    # ========================================================
    # SAVE REVIEW FILE
    # ========================================================

    output_path = (
        REPORTS_DIR /
        "repeat_validation_sample.csv"
    )

    sample.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    instructions_path = (
        REPORTS_DIR /
        "repeat_validation_instructions.md"
    )

    instructions = """# Repeat Detector Validation

## Sample

Review the 50 rows in:

`repeat_validation_sample.csv`

Each row contains:

- Current customer message
- Previous customer message
- Time gap
- Text similarity
- Product
- Category
- Previous product
- Previous category

## Label each row

Use exactly one of:

### TRUE

Use `TRUE` when the current contact appears to concern
the same underlying customer issue as the previous contact.

### FALSE

Use `FALSE` when the customer is contacting support about
a different issue, even if the customer/product/category is
the same.

### UNCLEAR

Use `UNCLEAR` when the available information is insufficient
to determine whether the issue is actually the same.

## Important

Do not label a ticket TRUE merely because:

- the customer is the same;
- the product is the same;
- the category is the same;
- the text sounds similar.

The underlying problem should be substantially the same.

## After reviewing

Save the CSV with your labels.

Then run:

`python src/04_validate_repeat_detection.py`

again.

The script will calculate:

- total reviewed;
- TRUE;
- FALSE;
- UNCLEAR;
- precision;
- error rate.
"""

    with open(
        instructions_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(instructions)

    # ========================================================
    # IF LABELS ALREADY EXIST, CALCULATE METRICS
    # ========================================================

    labels = sample["review_label"].astype(
        str
    ).str.strip().str.upper()

    reviewed = labels.isin(
        [
            "TRUE",
            "FALSE",
            "UNCLEAR"
        ]
    )

    if reviewed.sum() > 0:

        true_count = int(
            (labels == "TRUE").sum()
        )

        false_count = int(
            (labels == "FALSE").sum()
        )

        unclear_count = int(
            (labels == "UNCLEAR").sum()
        )

        decided_count = (
            true_count +
            false_count
        )

        if decided_count > 0:

            precision = (
                true_count /
                decided_count
                * 100
            )

            error_rate = (
                false_count /
                decided_count
                * 100
            )

            print("\nValidation results:")

            print(
                f"  TRUE: {true_count}"
            )

            print(
                f"  FALSE: {false_count}"
            )

            print(
                f"  UNCLEAR: {unclear_count}"
            )

            print(
                f"  Precision: {precision:.2f}%"
            )

            print(
                f"  Error rate: {error_rate:.2f}%"
            )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION SAMPLE CREATED")
    print("=" * 70)

    print(
        f"\n✓ {output_path}"
    )

    print(
        f"✓ {instructions_path}"
    )

    print(
        "\nOpen repeat_validation_sample.csv"
    )

    print(
        "and label each row TRUE, FALSE, or UNCLEAR."
    )

    print(
        "\nDo not calculate the final business goal yet."
    )


if __name__ == "__main__":
    main()
