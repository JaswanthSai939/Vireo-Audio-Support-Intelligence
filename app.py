from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="Vireo Audio Support Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def load_csv(filename):
    path = REPORTS_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not load {filename}: {exc}")
        return pd.DataFrame()


def load_json(filename):
    path = REPORTS_DIR / filename

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


@st.cache_data
def load_data():
    theme_metrics = load_csv(
        "complaint_theme_metrics.csv"
    )

    weekly_metrics = load_csv(
        "weekly_complaint_metrics.csv"
    )

    examples = load_csv(
        "complaint_examples.csv"
    )

    complaint_tickets = load_csv(
        "complaint_themed_tickets.csv"
    )

    repeat_candidates = load_csv(
        "repeat_candidates.csv"
    )

    return (
        theme_metrics,
        weekly_metrics,
        examples,
        complaint_tickets,
        repeat_candidates,
    )


@st.cache_data
def load_raw_tickets():
    path = DATA_DIR / "tickets.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_agents():
    path = DATA_DIR / "agents.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_business_goal():
    return load_json("business_goal.json")


@st.cache_data
def load_repeat_analysis():
    return load_json("repeat_analysis.json")


@st.cache_data
def load_business_analysis():
    return load_json("business_analysis.json")


# ============================================================
# LOAD
# ============================================================

(
    theme_metrics,
    weekly_metrics,
    examples,
    complaint_tickets,
    repeat_candidates,
) = load_data()

tickets = load_raw_tickets()
agents = load_agents()

business_goal = load_business_goal()
repeat_analysis = load_repeat_analysis()
business_analysis = load_business_analysis()


# ============================================================
# NORMALIZE DATA
# ============================================================

if not tickets.empty:
    if "created_at" in tickets.columns:
        tickets["created_at"] = pd.to_datetime(
            tickets["created_at"],
            errors="coerce",
        )

    if "resolved_at" in tickets.columns:
        tickets["resolved_at"] = pd.to_datetime(
            tickets["resolved_at"],
            errors="coerce",
        )

    if "csat_score" in tickets.columns:
        tickets["csat_score"] = pd.to_numeric(
            tickets["csat_score"],
            errors="coerce",
        )

    if "refund_amount_inr" in tickets.columns:
        tickets["refund_amount_inr"] = pd.to_numeric(
            tickets["refund_amount_inr"],
            errors="coerce",
        ).fillna(0)

    if "transfers" in tickets.columns:
        tickets["transfers"] = pd.to_numeric(
            tickets["transfers"],
            errors="coerce",
        ).fillna(0)

    # 0 means no CSAT response in this dataset.
    if "csat_score" in tickets.columns:
        tickets["valid_csat"] = tickets[
            "csat_score"
        ].mask(
            tickets["csat_score"] == 0
        )


if not weekly_metrics.empty:
    if "week" in weekly_metrics.columns:
        weekly_metrics["week"] = pd.to_datetime(
            weekly_metrics["week"],
            errors="coerce",
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎧 Vireo Audio")

st.sidebar.caption(
    "Support Intelligence Dashboard"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Summary",
        "Weekly Digest",
        "Complaint Themes",
        "Agent Leaderboard",
        "Repeat Contacts",
        "Evidence",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Data: Jan 2025 – Jun 2026"
)

st.sidebar.caption(
    "Unique analytical tickets: 11,875"
)


# ============================================================
# HEADER
# ============================================================

st.title("Vireo Audio Support Intelligence")

st.caption(
    "AI-assisted analysis of customer support tickets "
    "using the supplied Vireo Audio data pack."
)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.header("Executive Summary")

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_tickets = (
        len(complaint_tickets)
        if not complaint_tickets.empty
        else (
            len(tickets)
            if not tickets.empty
            else 0
        )
    )

    total_refunds = 0

    if not theme_metrics.empty:
        if "refund_amount_inr" in theme_metrics.columns:
            total_refunds = theme_metrics[
                "refund_amount_inr"
            ].sum()

    repeat_rate = None

    if isinstance(repeat_analysis, dict):
        repeat_rate = (
            repeat_analysis.get(
                "repeat_candidate_rate_percent"
            )
        )

    if repeat_rate is None and not repeat_candidates.empty:
        repeat_rate = (
            len(repeat_candidates)
            / total_tickets
            * 100
            if total_tickets
            else 0
        )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Tickets analyzed",
        f"{total_tickets:,}",
    )

    c2.metric(
        "Refund exposure",
        money(total_refunds),
    )

    if repeat_rate is not None:
        c3.metric(
            "Repeat-contact signal",
            f"{float(repeat_rate):.2f}%",
        )
    else:
        c3.metric(
            "Repeat-contact signal",
            "N/A",
        )

    if not theme_metrics.empty:
        top_theme = theme_metrics.iloc[0]["theme"]
    else:
        top_theme = "N/A"

    c4.metric(
        "Largest complaint theme",
        str(top_theme),
    )

    st.divider()

    # --------------------------------------------------------
    # BUSINESS GOAL
    # --------------------------------------------------------

    st.subheader("Business Goal")

    goal_text = None

    if isinstance(business_goal, dict):

        # Try several likely keys without assuming
        # a particular JSON structure.

        for key in [
            "recommended_goal",
            "goal_statement",
            "business_goal",
            "goal",
            "target",
        ]:
            value = business_goal.get(key)

            if isinstance(value, str) and value.strip():
                goal_text = value
                break

    if goal_text:
        st.info(goal_text)

    else:
        st.info(
            "Observed opportunity: reduce avoidable support "
            "contact/refund leakage. The exact target should "
            "be taken from the business-goal analysis report."
        )

    # --------------------------------------------------------
    # TOP THEMES
    # --------------------------------------------------------

    st.subheader("What customers are contacting support about")

    if not theme_metrics.empty:

        display_cols = [
            col
            for col in [
                "theme",
                "tickets",
                "ticket_share_percent",
                "avg_csat",
                "refund_amount_inr",
                "replacement_tickets",
                "transfer_tickets",
            ]
            if col in theme_metrics.columns
        ]

        table = theme_metrics[
            display_cols
        ].copy()

        rename_map = {
            "theme": "Complaint theme",
            "tickets": "Tickets",
            "ticket_share_percent": "Share %",
            "avg_csat": "CSAT",
            "refund_amount_inr": "Refund ₹",
            "replacement_tickets": "Replacements",
            "transfer_tickets": "Transfers",
        }

        table = table.rename(
            columns=rename_map
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning(
            "Complaint theme report is not available."
        )

    # --------------------------------------------------------
    # MAIN OBSERVATION
    # --------------------------------------------------------

    if not theme_metrics.empty:

        top = theme_metrics.iloc[0]

        st.subheader("Key observation")

        st.write(
            f"The largest discovered theme is "
            f"**{top['theme']}**, representing "
            f"**{top['ticket_share_percent']:.2f}%** "
            f"of analyzed tickets "
            f"({int(top['tickets']):,} tickets)."
        )

        if "avg_csat" in top and pd.notna(
            top["avg_csat"]
        ):
            st.write(
                f"Its average CSAT is "
                f"**{top['avg_csat']:.2f}**."
            )


# ============================================================
# WEEKLY DIGEST
# ============================================================

elif page == "Weekly Digest":

    st.header("Weekly Support Digest")

    st.caption(
        "Weekly complaint volume and associated support signals."
    )

    if weekly_metrics.empty:

        st.warning(
            "weekly_complaint_metrics.csv was not found."
        )

    else:

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        themes = sorted(
            weekly_metrics["theme"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_themes = st.multiselect(
            "Complaint themes",
            themes,
            default=themes[:6],
        )

        filtered = weekly_metrics.copy()

        if selected_themes:
            filtered = filtered[
                filtered["theme"].isin(
                    selected_themes
                )
            ]

        # ----------------------------------------------------
        # TOTAL WEEKLY TICKETS
        # ----------------------------------------------------

        weekly_total = (
            filtered.groupby("week")["tickets"]
            .sum()
            .sort_index()
        )

        st.subheader("Weekly ticket volume")

        if not weekly_total.empty:
            st.line_chart(
                weekly_total,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # WEEKLY THEME TABLE
        # ----------------------------------------------------

        st.subheader("Weekly complaint detail")

        display_cols = [
            col
            for col in [
                "week",
                "theme",
                "tickets",
                "avg_csat",
                "refund_amount_inr",
                "replacement_tickets",
                "transfer_tickets",
                "wow_change_percent",
            ]
            if col in filtered.columns
        ]

        weekly_table = filtered[
            display_cols
        ].sort_values(
            ["week", "tickets"],
            ascending=[False, False],
        )

        rename_map = {
            "week": "Week",
            "theme": "Theme",
            "tickets": "Tickets",
            "avg_csat": "CSAT",
            "refund_amount_inr": "Refund ₹",
            "replacement_tickets": "Replacements",
            "transfer_tickets": "Transfers",
            "wow_change_percent": "WoW %",
        }

        weekly_table = weekly_table.rename(
            columns=rename_map
        )

        st.dataframe(
            weekly_table,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # GENERATED DIGEST
    # --------------------------------------------------------

    digest_path = (
        REPORTS_DIR /
        "weekly_support_digest.md"
    )

    if digest_path.exists():

        st.divider()

        st.subheader(
            "Generated weekly digest"
        )

        digest = digest_path.read_text(
            encoding="utf-8"
        )

        st.markdown(digest)


# ============================================================
# COMPLAINT THEMES
# ============================================================

elif page == "Complaint Themes":

    st.header("Complaint Theme Explorer")

    if theme_metrics.empty:

        st.warning(
            "complaint_theme_metrics.csv was not found."
        )

    else:

        theme_list = (
            theme_metrics["theme"]
            .dropna()
            .tolist()
        )

        selected_theme = st.selectbox(
            "Select a complaint theme",
            theme_list,
        )

        selected = theme_metrics[
            theme_metrics["theme"]
            == selected_theme
        ]

        if not selected.empty:

            row = selected.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Tickets",
                f"{int(row['tickets']):,}",
            )

            c2.metric(
                "Share",
                f"{row['ticket_share_percent']:.2f}%",
            )

            c3.metric(
                "CSAT",
                (
                    f"{row['avg_csat']:.2f}"
                    if pd.notna(
                        row["avg_csat"]
                    )
                    else "N/A"
                ),
            )

            c4.metric(
                "Refund exposure",
                money(
                    row["refund_amount_inr"]
                ),
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Refund tickets",
                f"{int(row.get('refund_tickets', 0)):,}",
            )

            col2.metric(
                "Replacements",
                f"{int(row.get('replacement_tickets', 0)):,}",
            )

            col3.metric(
                "Transfer tickets",
                f"{int(row.get('transfer_tickets', 0)):,}",
            )

            # ------------------------------------------------
            # WEEKLY TREND FOR SELECTED THEME
            # ------------------------------------------------

            st.subheader(
                f"Weekly trend — {selected_theme}"
            )

            theme_weekly = weekly_metrics[
                weekly_metrics["theme"]
                == selected_theme
            ].sort_values("week")

            if not theme_weekly.empty:

                chart_data = theme_weekly[
                    ["week", "tickets"]
                ].set_index("week")

                st.line_chart(
                    chart_data,
                    use_container_width=True,
                )

            # ------------------------------------------------
            # EXAMPLES
            # ------------------------------------------------

            st.subheader(
                "Example customer messages"
            )

            if not examples.empty:

                theme_examples = examples[
                    examples["theme"]
                    == selected_theme
                ]

                if theme_examples.empty:

                    st.info(
                        "No examples available for this theme."
                    )

                else:

                    for _, example in (
                        theme_examples.head(5)
                        .iterrows()
                    ):

                        ticket_id = example.get(
                            "ticket_id",
                            "Unknown",
                        )

                        message = example.get(
                            "customer_message",
                            "",
                        )

                        with st.expander(
                            f"Ticket {ticket_id}"
                        ):
                            st.write(message)

                            if "category" in example:
                                st.caption(
                                    f"Category: "
                                    f"{example['category']}"
                                )

                            if "csat_score" in example:
                                st.caption(
                                    f"CSAT: "
                                    f"{example['csat_score']}"
                                )


# ============================================================
# AGENT LEADERBOARD
# ============================================================

elif page == "Agent Leaderboard":

    st.header("Agent Tickets Closed")

    st.warning(
        "Operational view only: raw ticket counts are not a "
        "fair productivity measure across teams with different "
        "case complexity. In particular, the client specifically "
        "asked not to rank the warranty team by ticket count."
    )

    if tickets.empty:

        st.error(
            "data/tickets.csv was not found."
        )

    else:

        leaderboard = tickets.copy()

        # ----------------------------------------------------
        # RESOLUTION DATE
        # ----------------------------------------------------

        if "resolved_at" not in leaderboard.columns:

            st.error(
                "resolved_at is not available in tickets.csv."
            )

        else:

            leaderboard = leaderboard[
                leaderboard["resolved_at"].notna()
            ].copy()

            # A ticket is counted as closed/resolved when
            # it has a resolution timestamp.

            leaderboard["week"] = (
                leaderboard["resolved_at"]
                .dt.to_period("W-SUN")
                .apply(
                    lambda x: x.start_time
                )
            )

            # ------------------------------------------------
            # TEAM INFORMATION
            # ------------------------------------------------

            if not agents.empty and "agent_id" in agents.columns:

                agents_copy = agents.copy()

                # Convert roster dates if present.
                for col in [
                    "from_date",
                    "to_date",
                ]:

                    if col in agents_copy.columns:
                        agents_copy[col] = pd.to_datetime(
                            agents_copy[col],
                            errors="coerce",
                        )

                # One row per agent is required for display.
                agent_info = (
                    agents_copy.sort_values(
                        [
                            col
                            for col in [
                                "agent_id",
                                "from_date",
                            ]
                            if col in agents_copy.columns
                        ]
                    )
                    .drop_duplicates(
                        "agent_id",
                        keep="last",
                    )
                )

                merge_cols = [
                    col
                    for col in [
                        "agent_id",
                        "name",
                        "site",
                        "team",
                        "shift",
                        "tier",
                    ]
                    if col in agent_info.columns
                ]

                leaderboard = leaderboard.merge(
                    agent_info[merge_cols],
                    on="agent_id",
                    how="left",
                )

            # ------------------------------------------------
            # FILTERS
            # ------------------------------------------------

            filter_col1, filter_col2 = st.columns(2)

            teams = []

            if "team" in leaderboard.columns:

                teams = sorted(
                    leaderboard["team"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

            selected_team = filter_col1.selectbox(
                "Team",
                ["All"] + teams,
            )

            available_weeks = sorted(
                leaderboard["week"]
                .dropna()
                .unique()
            )

            selected_week = filter_col2.selectbox(
                "Week",
                ["All"] + available_weeks,
            )

            filtered = leaderboard.copy()

            if selected_team != "All":
                filtered = filtered[
                    filtered["team"].astype(str)
                    == selected_team
                ]

            if selected_week != "All":
                filtered = filtered[
                    filtered["week"]
                    == selected_week
                ]

            # ------------------------------------------------
            # LEADERBOARD
            # ------------------------------------------------

            if "agent_id" in filtered.columns:

                group_cols = [
                    col
                    for col in [
                        "agent_id",
                        "name",
                        "team",
                        "site",
                        "shift",
                        "tier",
                    ]
                    if col in filtered.columns
                ]

                agent_counts = (
                    filtered.groupby(
                        group_cols,
                        dropna=False,
                    )
                    .size()
                    .reset_index(
                        name="tickets_closed"
                    )
                    .sort_values(
                        "tickets_closed",
                        ascending=False,
                    )
                )

                st.subheader(
                    "Tickets closed"
                )

                st.dataframe(
                    agent_counts,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # TEAM VOLUME
                # ------------------------------------------------

                if "team" in filtered.columns:

                    st.subheader(
                        "Tickets closed by team"
                    )

                    team_counts = (
                        filtered.groupby("team")
                        .size()
                        .sort_values(
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        team_counts,
                        use_container_width=True,
                    )

            else:

                st.warning(
                    "agent_id is not available."
                )


# ============================================================
# REPEAT CONTACTS
# ============================================================

elif page == "Repeat Contacts":

    st.header("Repeat Contact Analysis")

    st.caption(
        "Repeat-contact detection is treated as a candidate "
        "signal unless manually validated."
    )

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    if isinstance(repeat_analysis, dict):

        values = {}

        possible_keys = [
            "repeat_candidate_rate_percent",
            "strong_candidate_rate_percent",
            "repeat_candidates",
            "strong_repeat_candidates",
            "average_csat_candidates",
            "average_csat_other",
        ]

        for key in possible_keys:

            if key in repeat_analysis:
                values[key] = repeat_analysis[key]

        if values:

            c1, c2, c3, c4 = st.columns(4)

            if (
                "repeat_candidate_rate_percent"
                in values
            ):
                c1.metric(
                    "Candidate rate",
                    f"{float(values['repeat_candidate_rate_percent']):.2f}%",
                )

            if (
                "strong_candidate_rate_percent"
                in values
            ):
                c2.metric(
                    "Strong candidate rate",
                    f"{float(values['strong_candidate_rate_percent']):.2f}%",
                )

            if (
                "average_csat_candidates"
                in values
            ):
                c3.metric(
                    "Candidate CSAT",
                    f"{float(values['average_csat_candidates']):.2f}",
                )

            if (
                "average_csat_other"
                in values
            ):
                c4.metric(
                    "Other-ticket CSAT",
                    f"{float(values['average_csat_other']):.2f}",
                )

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    if not repeat_candidates.empty:

        st.subheader(
            "Repeat-contact candidates"
        )

        st.dataframe(
            repeat_candidates.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Candidate detection should not be presented as a "
            "validated KPI without manual validation."
        )

    else:

        st.info(
            "repeat_candidates.csv was not found."
        )


# ============================================================
# EVIDENCE
# ============================================================

elif page == "Evidence":

    st.header("Evidence & Audit Trail")

    st.write(
        "This page shows the evidence behind the dashboard "
        "rather than only presenting summary numbers."
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    st.subheader("Generated analysis files")

    report_files = sorted(
        [
            p.name
            for p in REPORTS_DIR.iterdir()
            if p.is_file()
        ]
    )

    if report_files:

        for filename in report_files:
            st.write(f"• `{filename}`")

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.subheader("Data quality")

    if not tickets.empty:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Raw ticket rows",
            f"{len(tickets):,}",
        )

        if "ticket_id" in tickets.columns:

            duplicates = tickets[
                "ticket_id"
            ].duplicated().sum()

        else:
            duplicates = 0

        c2.metric(
            "Duplicate ticket IDs",
            f"{duplicates:,}",
        )

        if "customer_message" in tickets.columns:

            empty_messages = (
                tickets[
                    "customer_message"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

        else:
            empty_messages = 0

        c3.metric(
            "Empty customer messages",
            f"{empty_messages:,}",
        )

    # --------------------------------------------------------
    # COMPLAINT EXAMPLES
    # --------------------------------------------------------

    if not examples.empty:

        st.subheader(
            "Complaint examples used by the analysis"
        )

        st.dataframe(
            examples,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.info(
        "The complaint themes are machine-discovered using "
        "TF-IDF and clustering. Theme names are interpretations "
        "of cluster terms, so individual tickets should be "
        "reviewed before treating a theme as a definitive "
        "customer-intent category."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Vireo Audio — Support Intelligence"
)

st.sidebar.caption(
    "Built from the supplied assessment data pack."
)
