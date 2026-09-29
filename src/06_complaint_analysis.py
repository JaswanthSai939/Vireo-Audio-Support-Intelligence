from pathlib import Path

import json

import re

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans


# ============================================================

# VIREO AUDIO

# STEP 6: AI-ASSISTED COMPLAINT THEME ANALYSIS

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================

# CONFIGURATION

# ============================================================

N_CLUSTERS = 12

MIN_DF = 5

MAX_DF = 0.90

TOP_TERMS_PER_THEME = 8

RANDOM_STATE = 42


# ============================================================

# LOAD DATA

# ============================================================

def load_tickets():

    path = DATA_DIR / "tickets.csv"

    if not path.exists():

        raise FileNotFoundError(

            f"Could not find {path}"

        )

    tickets = pd.read_csv(path)

    print(

        f"Raw tickets: {len(tickets):,}"

    )

    return tickets


# ============================================================

# DEDUPLICATION

# ============================================================

def deduplicate_tickets(tickets):

    tickets = tickets.copy()

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

    duplicate_count = (

        tickets["ticket_id"]

        .duplicated()

        .sum()

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

    print(

        f"Duplicate rows removed: {duplicate_count:,}"

    )

    print(

        f"Unique tickets: {len(tickets):,}"

    )

    return tickets


# ============================================================

# TEXT CLEANING

# ============================================================

def clean_text(text):

    if pd.isna(text):

        return ""

    text = str(text).lower()

    # Remove URLs.

    text = re.sub(

        r"https?://\S+|www.\S+",

        " ",

        text

    )

    # Remove email addresses.

    text = re.sub(

        r"\S+@\S+",

        " ",

        text

    )

    # Remove ticket/order IDs and long numbers.

    text = re.sub(

        r"\b[a-z]*\d{4,}[a-z0-9-]*\b",

        " ",

        text

    )

    # Keep letters and spaces.

    text = re.sub(

        r"[^a-z\s]",

        " ",

        text

    )

    # Normalize whitespace.

    text = re.sub(

        r"\s+",

        " ",

        text

    )

    return text.strip()


# ============================================================

# BUILD TEXT CORPUS

# ============================================================

def build_text_corpus(tickets):

    tickets = tickets.copy()

    tickets["clean_message"] = (

        tickets["customer_message"]

        .apply(clean_text)

    )

    # Remove empty messages.

    empty_count = (

        tickets["clean_message"]

        .str.len()

        .eq(0)

        .sum()

    )

    print(

        f"Empty customer messages: {empty_count:,}"

    )

    return tickets


# ============================================================

# TF-IDF

# ============================================================

def build_tfidf(tickets):

    print(

        "\nBuilding TF-IDF representation..."

    )

    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2),

        min_df=MIN_DF,

        max_df=MAX_DF,

        sublinear_tf=True,

        max_features=15000

    )

    matrix = vectorizer.fit_transform(

        tickets["clean_message"]

    )

    print(

        f"TF-IDF matrix: "

        f"{matrix.shape[0]:,} documents x "

        f"{matrix.shape[1]:,} terms"

    )

    return vectorizer, matrix


# ============================================================

# K-MEANS CLUSTERING

# ============================================================

def cluster_tickets(matrix):

    print(

        f"\nDiscovering {N_CLUSTERS} complaint themes..."

    )

    model = KMeans(

        n_clusters=N_CLUSTERS,

        random_state=RANDOM_STATE,

        n_init=10

    )

    labels = model.fit_predict(

        matrix

    )

    return model, labels


# ============================================================

# EXTRACT TOP TERMS

# ============================================================

def get_cluster_terms(

    model,

    vectorizer

):

    feature_names = (

        vectorizer

        .get_feature_names_out()

    )

    cluster_terms = {}

    for cluster_id in range(

        N_CLUSTERS

    ):

        center = model.cluster_centers_[

            cluster_id

        ]

        top_indices = center.argsort()[

            ::-1

        ][

            :TOP_TERMS_PER_THEME

        ]

        terms = [

            feature_names[index]

            for index in top_indices

        ]

        cluster_terms[

            cluster_id

        ] = terms

    return cluster_terms


# ============================================================

# AUTOMATIC THEME NAME

# ============================================================

def name_theme(

    terms

):

    text = " ".join(

        terms

    )

    rules = [

        (

            [

                "battery",

                "charging",

                "charge",

                "charging case",

                "power"

            ],

            "Charging / Battery"

        ),

        (

            [

                "bluetooth",

                "pairing",

                "pair",

                "connect",

                "connection",

                "disconnect"

            ],

            "Bluetooth / Connectivity"

        ),

        (

            [

                "delivery",

                "delivered",

                "shipping",

                "shipped",

                "courier",

                "package",

                "order"

            ],

            "Delivery / Order Issues"

        ),

        (

            [

                "refund",

                "refunded",

                "money",

                "refund pending"

            ],

            "Refunds"

        ),

        (

            [

                "return",

                "pickup",

                "returning",

                "replacement"

            ],

            "Returns / Replacements"

        ),

        (

            [

                "invoice",

                "gst",

                "billing",

                "bill",

                "payment"

            ],

            "Billing / Payment"

        ),

        (

            [

                "discount",

                "coupon",

                "promo",

                "offer",

                "voucher"

            ],

            "Promotions / Discounts"

        ),

        (

            [

                "microphone",

                "mic",

                "voice",

                "call",

                "hear"

            ],

            "Microphone / Call Quality"

        ),

        (

            [

                "sound",

                "audio",

                "noise",

                "volume",

                "hiss",

                "distortion"

            ],

            "Audio Quality"

        ),

        (

            [

                "screen",

                "display",

                "touch"

            ],

            "Screen / Display"

        ),

        (

            [

                "app",

                "login",

                "account",

                "password",

                "otp"

            ],

            "App / Account"

        ),

        (

            [

                "strap",

                "band",

                "watch",

                "watch face"

            ],

            "Watch / Physical Damage"

        ),

    ]

    for keywords, name in rules:

        matches = sum(

            keyword in text

            for keyword in keywords

        )

        if matches >= 1:

            return name

    # Fallback: use first few cluster terms.

    return " / ".join(

        terms[:3]

    ).title()


# ============================================================

# ASSIGN THEMES

# ============================================================

def create_theme_table(

    tickets,

    labels,

    cluster_terms

):

    tickets = tickets.copy()

    tickets["cluster_id"] = labels

    tickets["theme_terms"] = (

        tickets["cluster_id"]

        .map(

            lambda x:

            ", ".join(

                cluster_terms[x]

            )

        )

    )

    tickets["theme"] = (

        tickets["cluster_id"]

        .map(

            lambda x:

            name_theme(

                cluster_terms[x]

            )

        )

    )

    return tickets


# ============================================================

# BUSINESS METRICS

# ============================================================

def calculate_theme_metrics(

    tickets

):

    data = tickets.copy()

    data["refund_amount_inr"] = pd.to_numeric(

        data["refund_amount_inr"],

        errors="coerce"

    ).fillna(0)

    data["transfers"] = pd.to_numeric(

        data["transfers"],

        errors="coerce"

    ).fillna(0)

    data["csat_score"] = pd.to_numeric(

        data["csat_score"],

        errors="coerce"

    )

    data["replacement_flag"] = (

        data["replacement_issued"]

        .astype(str)

        .str.upper()

        .eq("Y")

    )

    # CSAT 0 means no response.

    data["valid_csat"] = data[

        "csat_score"

    ].mask(

        data["csat_score"] == 0

    )

    grouped = (

        data.groupby("theme")

        .agg(

            tickets=(

                "ticket_id",

                "count"

            ),

            avg_csat=(

                "valid_csat",

                "mean"

            ),

            refund_tickets=(

                "refund_amount_inr",

                lambda x:

                (x > 0).sum()

            ),

            refund_amount_inr=(

                "refund_amount_inr",

                "sum"

            ),

            replacement_tickets=(

                "replacement_flag",

                "sum"

            ),

            transfer_tickets=(

                "transfers",

                lambda x:

                (x > 0).sum()

            ),

            total_transfers=(

                "transfers",

                "sum"

            ),

        )

        .reset_index()

    )

    grouped["ticket_share_percent"] = (

        grouped["tickets"]

        / len(data)

        * 100

    )

    grouped["refund_rate_percent"] = (

        grouped["refund_tickets"]

        / grouped["tickets"]

        * 100

    )

    grouped["replacement_rate_percent"] = (

        grouped["replacement_tickets"]

        / grouped["tickets"]

        * 100

    )

    grouped["transfer_rate_percent"] = (

        grouped["transfer_tickets"]

        / grouped["tickets"]

        * 100

    )

    grouped = grouped.sort_values(

        "tickets",

        ascending=False

    )

    return grouped


# ============================================================

# WEEKLY TREND

# ============================================================

# ============================================================

# WEEKLY TREND

# ============================================================

def calculate_weekly_metrics(

    tickets

):

    data = tickets.copy()

    # --------------------------------------------------------

    # Create business-analysis fields locally.

    # These must exist before groupby().agg().

    # --------------------------------------------------------

    data["created_at"] = pd.to_datetime(

        data["created_at"],

        errors="coerce"

    )

    data["refund_amount_inr"] = pd.to_numeric(

        data["refund_amount_inr"],

        errors="coerce"

    ).fillna(0)

    data["transfers"] = pd.to_numeric(

        data["transfers"],

        errors="coerce"

    ).fillna(0)

    data["csat_score"] = pd.to_numeric(

        data["csat_score"],

        errors="coerce"

    )

    # CSAT = 0 means no response.

    data["valid_csat"] = data[

        "csat_score"

    ].mask(

        data["csat_score"] == 0

    )

    # Replacement flag.

    data["replacement_flag"] = (

        data["replacement_issued"]

        .astype(str)

        .str.strip()

        .str.upper()

        .eq("Y")

    )

    # Remove rows without valid creation date.

    data = data[

        data["created_at"].notna()

    ].copy()

    # --------------------------------------------------------

    # Create Monday-based weekly periods.

    # --------------------------------------------------------

    data["week"] = (

        data["created_at"]

        .dt.to_period("W-SUN")

        .apply(

            lambda x: x.start_time

        )

    )

    # --------------------------------------------------------

    # Aggregate weekly theme metrics.

    # --------------------------------------------------------

    weekly = (

        data.groupby(

            [

                "week",

                "theme"

            ]

        )

        .agg(

            tickets=(

                "ticket_id",

                "count"

            ),

            avg_csat=(

                "valid_csat",

                "mean"

            ),

            refund_amount_inr=(

                "refund_amount_inr",

                "sum"

            ),

            replacement_tickets=(

                "replacement_flag",

                "sum"

            ),

            transfer_tickets=(

                "transfers",

                lambda x:

                (x > 0).sum()

            ),

        )

        .reset_index()

    )

    # --------------------------------------------------------

    # Sort before calculating week-over-week change.

    # --------------------------------------------------------

    weekly = weekly.sort_values(

        [

            "theme",

            "week"

        ]

    ).reset_index(

        drop=True

    )

    # --------------------------------------------------------

    # Previous week's ticket count.

    # --------------------------------------------------------

    weekly["previous_week_tickets"] = (

        weekly

        .groupby("theme")[

            "tickets"

        ]

        .shift(1)

    )

    # --------------------------------------------------------

    # Week-over-week percentage change.

    # --------------------------------------------------------

    weekly["wow_change_percent"] = np.where(

        weekly["previous_week_tickets"].gt(0),

        (

            (

                weekly["tickets"]

                -

                weekly["previous_week_tickets"]

            )

            /

            weekly["previous_week_tickets"]

            *

            100

        ),

        np.nan

    )

    return weekly


# ============================================================

# TOP EXAMPLES

# ============================================================

def extract_examples(

    tickets,

    metrics,

    number_of_themes=8,

    examples_per_theme=3

):

    examples = []

    top_themes = metrics.head(

        number_of_themes

    )["theme"].tolist()

    for theme in top_themes:

        subset = tickets[

            tickets["theme"] == theme

        ].copy()

        subset = subset[

            subset["customer_message"]

            .notna()

        ]

        subset = subset.head(

            examples_per_theme

        )

        for _, row in subset.iterrows():

            examples.append({

                "theme": theme,

                "ticket_id": row[

                    "ticket_id"

                ],

                "customer_message": row[

                    "customer_message"

                ],

                "category": row.get(

                    "category",

                    ""

                ),

                "product_sku": row.get(

                    "product_sku",

                    ""

                ),

                "csat_score": row.get(

                    "csat_score",

                    ""

                ),

            })

    return pd.DataFrame(

        examples

    )


# ============================================================

# WEEKLY DIGEST

# ============================================================

def generate_digest(
    tickets,
    metrics,
    weekly
):
    # Ensure created_at is datetime before using date/strftime operations.
    data = tickets.copy()

    data["created_at"] = pd.to_datetime(
        data["created_at"],
        errors="coerce"
    )

    valid_dates = data["created_at"].dropna()

    if len(valid_dates) > 0:
        earliest_date = valid_dates.min()
        latest_date = valid_dates.max()
    else:
        earliest_date = None
        latest_date = None

    latest_week = weekly["week"].max()

    current_week = weekly[
        weekly["week"] == latest_week
    ].copy()

    current_week = current_week.sort_values(
        "tickets",
        ascending=False
    )

    lines = []

    lines.append("# Vireo Audio — Weekly Support Digest")
    lines.append("")

    if earliest_date is not None:
        lines.append(
            f"Data coverage: "
            f"{earliest_date.strftime('%Y-%m-%d')} "
            f"to "
            f"{latest_date.strftime('%Y-%m-%d')}"
        )
    else:
        lines.append("Data coverage: unavailable")

    lines.append("")

    if pd.notna(latest_week):
        lines.append(
            f"Latest available week: "
            f"{pd.Timestamp(latest_week).strftime('%Y-%m-%d')}"
        )
    else:
        lines.append("Latest available week: unavailable")

    lines.append("")
    lines.append("## Top Complaint Themes")
    lines.append("")
    lines.append(
        "| Theme | Tickets | Share | WoW | CSAT | Refunds | Replacements | Transfers |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    top_themes = metrics.head(8)

    for _, row in top_themes.iterrows():
        theme = row["theme"]
        current = current_week[current_week["theme"] == theme]

        wow = (
            current.iloc[0]["wow_change_percent"]
            if len(current) > 0
            else np.nan
        )

        wow_text = f"{wow:+.1f}%" if pd.notna(wow) else "N/A"

        csat = row["avg_csat"]
        csat_text = f"{csat:.2f}" if pd.notna(csat) else "N/A"

        lines.append(
            f"| {theme} | "
            f"{int(row['tickets']):,} | "
            f"{row['ticket_share_percent']:.1f}% | "
            f"{wow_text} | "
            f"{csat_text} | "
            f"₹{row['refund_amount_inr']:,.0f} | "
            f"{int(row['replacement_tickets']):,} | "
            f"{int(row['transfer_tickets']):,} |"
        )

    lines.append("")
    lines.append("## Signals to Investigate")
    lines.append("")

    if len(metrics) > 0:
        highest_volume = metrics.iloc[0]
        lines.append(
            f"- **Highest-volume theme:** "
            f"{highest_volume['theme']} "
            f"({int(highest_volume['tickets']):,} tickets)."
        )

        csat_candidates = metrics[
            metrics["tickets"] >= 30
        ].dropna(subset=["avg_csat"])

        if len(csat_candidates) > 0:
            lowest_csat = csat_candidates.sort_values(
                "avg_csat"
            ).iloc[0]
            lines.append(
                f"- **Lowest-CSAT material theme:** "
                f"{lowest_csat['theme']} "
                f"({lowest_csat['avg_csat']:.2f} average CSAT)."
            )

        refund_theme = metrics.sort_values(
            "refund_amount_inr",
            ascending=False
        ).iloc[0]
        lines.append(
            f"- **Highest refund exposure:** "
            f"{refund_theme['theme']} "
            f"({money(refund_theme['refund_amount_inr'])})."
        )

        transfer_candidates = metrics[
            metrics["tickets"] >= 30
        ]

        if len(transfer_candidates) > 0:
            transfer_theme = transfer_candidates.sort_values(
                "transfer_rate_percent",
                ascending=False
            ).iloc[0]
            lines.append(
                f"- **Highest transfer-rate material theme:** "
                f"{transfer_theme['theme']} "
                f"({transfer_theme['transfer_rate_percent']:.1f}%)."
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "These are observational signals from historical support "
        "tickets. Association with refunds, replacements or "
        "transfers does not establish causation."
    )

    lines.append("")
    lines.append("## Data Quality")
    lines.append("")
    lines.append(f"- Unique tickets analyzed: {len(tickets):,}")
    lines.append(f"- Discovered complaint themes: {len(metrics):,}")
    lines.append("- Theme discovery method: TF-IDF + KMeans clustering")
    lines.append("- No paid LLM/API calls used")

    return "\n".join(lines)


def money(value):

    return f"₹{value:,.0f}"


# ============================================================

# MAIN

# ============================================================

def main():

    print("=" * 70)

    print(

        "VIREO AUDIO - STEP 6: "

        "AI-ASSISTED COMPLAINT ANALYSIS"

    )

    print("=" * 70)

    # --------------------------------------------------------

    # 1. Load

    # --------------------------------------------------------

    print("\n[1/7] Loading tickets...")

    tickets = load_tickets()

    # --------------------------------------------------------

    # 2. Deduplicate

    # --------------------------------------------------------

    print("\n[2/7] Deduplicating...")

    tickets = deduplicate_tickets(

        tickets

    )

    # --------------------------------------------------------

    # 3. Prepare text

    # --------------------------------------------------------

    print("\n[3/7] Preparing customer messages...")

    tickets = build_text_corpus(

        tickets

    )

    # --------------------------------------------------------

    # 4. TF-IDF

    # --------------------------------------------------------

    print("\n[4/7] Building NLP representation...")

    vectorizer, matrix = build_tfidf(

        tickets

    )

    # --------------------------------------------------------

    # 5. Clustering

    # --------------------------------------------------------

    print("\n[5/7] Discovering complaint themes...")

    model, labels = cluster_tickets(

        matrix

    )

    cluster_terms = get_cluster_terms(

        model,

        vectorizer

    )

    tickets = create_theme_table(

        tickets,

        labels,

        cluster_terms

    )

    # --------------------------------------------------------

    # 6. Metrics

    # --------------------------------------------------------

    print("\n[6/7] Calculating business metrics...")

    metrics = calculate_theme_metrics(

        tickets

    )

    weekly = calculate_weekly_metrics(

        tickets

    )

    examples = extract_examples(

        tickets,

        metrics

    )

    # --------------------------------------------------------

    # Print themes

    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("DISCOVERED COMPLAINT THEMES")

    print("=" * 70)

    for _, row in metrics.iterrows():

        print(

            f"\n{row['theme']}"

        )

        print(

            f"  Tickets: "

            f"{int(row['tickets']):,}"

        )

        print(

            f"  Share: "

            f"{row['ticket_share_percent']:.2f}%"

        )

        print(

            f"  CSAT: "

            f"{row['avg_csat']:.2f}"

            if pd.notna(row["avg_csat"])

            else "  CSAT: N/A"

        )

        print(

            f"  Refund amount: "

            f"{money(row['refund_amount_inr'])}"

        )

        print(

            f"  Replacements: "

            f"{int(row['replacement_tickets']):,}"

        )

        print(

            f"  Transfers: "

            f"{int(row['transfer_tickets']):,}"

        )

    # --------------------------------------------------------

    # Save enriched tickets

    # --------------------------------------------------------

    print("\n[7/7] Saving reports...")

    enriched_path = (

        REPORTS_DIR /

        "complaint_themed_tickets.csv"

    )

    tickets[

        [

            "ticket_id",

            "created_at",

            "customer_id",

            "product_sku",

            "category",

            "customer_message",

            "csat_score",

            "refund_amount_inr",

            "replacement_issued",

            "transfers",

            "theme",

            "cluster_id",

            "theme_terms"

        ]

    ].to_csv(

        enriched_path,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------

    # Save theme metrics

    # --------------------------------------------------------

    metrics_path = (

        REPORTS_DIR /

        "complaint_theme_metrics.csv"

    )

    metrics.to_csv(

        metrics_path,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------

    # Save weekly metrics

    # --------------------------------------------------------

    weekly_path = (

        REPORTS_DIR /

        "weekly_complaint_metrics.csv"

    )

    weekly.to_csv(

        weekly_path,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------

    # Save examples

    # --------------------------------------------------------

    examples_path = (

        REPORTS_DIR /

        "complaint_examples.csv"

    )

    examples.to_csv(

        examples_path,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------

    # Save digest

    # --------------------------------------------------------

    digest = generate_digest(

        tickets,

        metrics,

        weekly

    )

    digest_path = (

        REPORTS_DIR /

        "weekly_support_digest.md"

    )

    with open(

        digest_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            digest

        )

    # --------------------------------------------------------

    # Save JSON summary

    # --------------------------------------------------------

    summary = {

        "method": {

            "vectorization": "TF-IDF",

            "clustering": "KMeans",

            "n_clusters": N_CLUSTERS,

            "random_state": RANDOM_STATE,

        },

        "population": {

            "tickets": int(len(tickets))

        },

        "themes": metrics.to_dict(

            orient="records"

        ),

        "cluster_terms": {

            str(key): value

            for key, value

            in cluster_terms.items()

        },

        "limitations": [

            "Unsupervised clusters require human interpretation.",

            "Theme labels are heuristic names generated from top terms.",

            "Historical association does not prove causation.",

            "No paid LLM/API calls were used.",

            "Some messages may contain multiple issues."

        ]

    }

    json_path = (

        REPORTS_DIR /

        "complaint_analysis.json"

    )

    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            ensure_ascii=False,

            default=str

        )

    # --------------------------------------------------------

    # Final output

    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("STEP 6 COMPLETE")

    print("=" * 70)

    print(

        f"\n✓ Created: {enriched_path}"

    )

    print(

        f"✓ Created: {metrics_path}"

    )

    print(

        f"✓ Created: {weekly_path}"

    )

    print(

        f"✓ Created: {examples_path}"

    )

    print(

        f"✓ Created: {digest_path}"

    )

    print(

        f"✓ Created: {json_path}"

    )

    print(

        "\nNext step:"

    )

    print(

        "Review the discovered themes and weekly digest "

        "before building the Streamlit interface."

    )


if __name__ == "__main__":

    main()
