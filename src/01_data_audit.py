from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# Project root = vireo_assessment/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(filename):
    """Load a CSV file from the data directory."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def main():

    print("=" * 70)
    print("VIREO AUDIO - STEP 1: DATA AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\n[1/6] Loading datasets...")

    tickets = load_csv("tickets.csv")
    agents = load_csv("agents.csv")
    customers = load_csv("customers.csv")
    orders = load_csv("orders.csv")
    products = load_csv("products.csv")

    print("✓ tickets.csv loaded")
    print("✓ agents.csv loaded")
    print("✓ customers.csv loaded")
    print("✓ orders.csv loaded")
    print("✓ products.csv loaded")

    # --------------------------------------------------------
    # Basic dataset sizes
    # --------------------------------------------------------

    print("\n[2/6] Checking dataset sizes...")

    dataset_sizes = {
        "tickets": len(tickets),
        "agents": len(agents),
        "customers": len(customers),
        "orders": len(orders),
        "products": len(products),
    }

    for name, count in dataset_sizes.items():
        print(f"  {name:<12}: {count:,} rows")

    # --------------------------------------------------------
    # Ticket date range
    # --------------------------------------------------------

    print("\n[3/6] Checking ticket dates...")

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce"
    )

    min_date = tickets["created_at"].min()
    max_date = tickets["created_at"].max()

    print(f"  Earliest ticket : {min_date}")
    print(f"  Latest ticket   : {max_date}")

    # --------------------------------------------------------
    # Duplicate ticket IDs
    # --------------------------------------------------------

    print("\n[4/6] Checking duplicate ticket IDs...")

    raw_ticket_rows = len(tickets)

    unique_ticket_ids = tickets["ticket_id"].nunique()

    duplicate_ticket_rows = tickets["ticket_id"].duplicated(
        keep=False
    ).sum()

    duplicate_ticket_ids = (
        tickets.loc[
            tickets["ticket_id"].duplicated(keep=False),
            "ticket_id"
        ].nunique()
    )

    print(f"  Raw ticket rows       : {raw_ticket_rows:,}")
    print(f"  Unique ticket IDs     : {unique_ticket_ids:,}")
    print(f"  Duplicate ticket IDs  : {duplicate_ticket_ids:,}")
    print(f"  Rows involved in dupes: {duplicate_ticket_rows:,}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n[5/6] Checking important missing values...")

    important_columns = [
        "ticket_id",
        "created_at",
        "first_response_at",
        "resolved_at",
        "status",
        "channel",
        "customer_id",
        "product_sku",
        "category",
        "priority",
        "agent_id",
        "csat_score",
        "customer_message",
        "agent_notes",
    ]

    missing_values = {}

    for column in important_columns:

        if column in tickets.columns:
            missing_count = int(tickets[column].isna().sum())

            missing_values[column] = missing_count

            print(
                f"  {column:<22}: "
                f"{missing_count:,} missing"
            )

    # --------------------------------------------------------
    # Referential integrity checks
    # --------------------------------------------------------

    print("\n[6/6] Checking relationships between datasets...")

    # Customer IDs
    valid_customer_ids = set(
        customers["customer_id"].dropna()
    )

    ticket_customer_ids = set(
        tickets["customer_id"].dropna()
    )

    unknown_customer_ids = (
        ticket_customer_ids - valid_customer_ids
    )

    # Product SKUs
    valid_product_skus = set(
        products["sku"].dropna()
    )

    ticket_product_skus = set(
        tickets["product_sku"].dropna()
    )

    unknown_product_skus = (
        ticket_product_skus - valid_product_skus
    )

    # Agent IDs
    valid_agent_ids = set(
        agents["agent_id"].dropna()
    )

    ticket_agent_ids = set(
        tickets["agent_id"].dropna()
    )

    unknown_agent_ids = (
        ticket_agent_ids - valid_agent_ids
    )

    # Order IDs
    valid_order_ids = set(
        orders["order_id"].dropna()
    )

    ticket_order_ids = set(
        tickets["order_id"].dropna()
    )

    unknown_order_ids = (
        ticket_order_ids - valid_order_ids
    )

    print(
        f"  Unknown customer IDs : "
        f"{len(unknown_customer_ids):,}"
    )

    print(
        f"  Unknown product SKUs  : "
        f"{len(unknown_product_skus):,}"
    )

    print(
        f"  Unknown agent IDs     : "
        f"{len(unknown_agent_ids):,}"
    )

    print(
        f"  Unknown order IDs     : "
        f"{len(unknown_order_ids):,}"
    )

    # ========================================================
    # ADDITIONAL QUALITY CHECKS
    # ========================================================

    print("\n" + "=" * 70)
    print("ADDITIONAL QUALITY CHECKS")
    print("=" * 70)

    # Status values
    status_values = sorted(
        tickets["status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print("\nTicket statuses:")
    for value in status_values:
        count = int(
            (tickets["status"].astype(str) == value).sum()
        )
        print(f"  {value:<15}: {count:,}")

    # Channels
    channel_values = sorted(
        tickets["channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print("\nChannels:")
    for value in channel_values:
        count = int(
            (tickets["channel"].astype(str) == value).sum()
        )
        print(f"  {value:<15}: {count:,}")

    # Source systems
    source_values = sorted(
        tickets["source_system"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print("\nSource systems:")
    for value in source_values:
        count = int(
            (tickets["source_system"].astype(str) == value).sum()
        )
        print(f"  {value:<15}: {count:,}")

    # CSAT
    csat_numeric = pd.to_numeric(
        tickets["csat_score"],
        errors="coerce"
    )

    print("\nCSAT:")
    print(f"  Missing/blank CSAT : {csat_numeric.isna().sum():,}")
    print(f"  CSAT value = 0     : {(csat_numeric == 0).sum():,}")
    print(f"  Valid CSAT 1-5     : {csat_numeric.between(1, 5).sum():,}")

    # ========================================================
    # BUILD REPORT DATA
    # ========================================================

    audit_report = {
        "dataset_sizes": dataset_sizes,

        "tickets": {
            "raw_rows": raw_ticket_rows,
            "unique_ticket_ids": int(unique_ticket_ids),
            "duplicate_ticket_ids": int(duplicate_ticket_ids),
            "duplicate_ticket_rows": int(duplicate_ticket_rows),
            "earliest_created_at": (
                min_date.isoformat()
                if pd.notna(min_date)
                else None
            ),
            "latest_created_at": (
                max_date.isoformat()
                if pd.notna(max_date)
                else None
            ),
        },

        "missing_values": missing_values,

        "referential_integrity": {
            "unknown_customer_ids": len(unknown_customer_ids),
            "unknown_product_skus": len(unknown_product_skus),
            "unknown_agent_ids": len(unknown_agent_ids),
            "unknown_order_ids": len(unknown_order_ids),
        },

        "status_values": status_values,
        "channel_values": channel_values,
        "source_systems": source_values,

        "csat": {
            "missing_or_blank": int(csat_numeric.isna().sum()),
            "zero_values": int((csat_numeric == 0).sum()),
            "valid_1_to_5": int(csat_numeric.between(1, 5).sum()),
        },

        "analysis_decisions": [
            "Deduplicate ticket IDs before calculating ticket-level metrics.",
            "Investigate duplicated IDs because the support documentation describes legacy migration and re-imports.",
            "Treat legacy CSAT value 0 as no response rather than a zero satisfaction score.",
            "Exclude missing CSAT responses from CSAT averages.",
            "Validate joins against customers, products, agents and orders before business analysis."
        ]
    }

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    json_path = REPORTS_DIR / "audit_report.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            audit_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE MARKDOWN REPORT
    # ========================================================

    md_path = REPORTS_DIR / "01_data_audit.md"

    md_content = f"""# Vireo Audio — Data Audit Report

## 1. Dataset Overview

| Dataset | Rows |
|---|---:|
| Tickets | {dataset_sizes["tickets"]:,} |
| Agents | {dataset_sizes["agents"]:,} |
| Customers | {dataset_sizes["customers"]:,} |
| Orders | {dataset_sizes["orders"]:,} |
| Products | {dataset_sizes["products"]:,} |

## 2. Ticket Coverage

- Raw ticket rows: **{raw_ticket_rows:,}**
- Unique ticket IDs: **{unique_ticket_ids:,}**
- Duplicate ticket IDs: **{duplicate_ticket_ids:,}**
- Rows involved in duplicate IDs: **{duplicate_ticket_rows:,}**
- Earliest ticket: **{min_date}**
- Latest ticket: **{max_date}**

## 3. Referential Integrity

| Relationship | Unknown IDs |
|---|---:|
| Customer | {len(unknown_customer_ids):,} |
| Product | {len(unknown_product_skus):,} |
| Agent | {len(unknown_agent_ids):,} |
| Order | {len(unknown_order_ids):,} |

## 4. Important Data Decisions

1. Ticket-level analysis will use unique `ticket_id` records.
2. Duplicate ticket IDs will be investigated because the supplied context describes legacy migration and re-imports.
3. Legacy CSAT value `0` will be treated as no response.
4. Missing CSAT responses will be excluded from CSAT averages.
5. Referential integrity will be checked before joining datasets.

## 5. Initial Conclusion

The ticket export contains duplicate ticket IDs, so raw row counts should not be treated as ticket counts without investigation.

The next analysis stage will examine the cleaned ticket population for business-impact opportunities including repeat contacts, SLA breaches, transfers, refunds, replacements and customer-satisfaction patterns.
"""

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(md_content)

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA AUDIT COMPLETE")
    print("=" * 70)

    print(f"\n✓ Created: {json_path}")
    print(f"✓ Created: {md_path}")

    print("\nNext step:")
    print("Review the generated reports before moving to business analysis.")


if __name__ == "__main__":
    main()
