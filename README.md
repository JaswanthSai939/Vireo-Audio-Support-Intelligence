# Vireo Audio Support Intelligence

An AI-assisted customer-support analytics solution built from the supplied Vireo Audio support data pack.

## Objective

The solution analyzes historical support tickets to identify operational and customer-experience signals, including:

- Data quality and duplicate-ticket issues
- Support contact and financial exposure
- SLA breaches and internal transfers
- Refund and replacement exposure
- Repeat-contact candidates
- Complaint themes discovered from customer messages
- Weekly support trends and digest
- Agent ticket-volume views
- Evidence and auditability through generated reports

The dashboard is intended to help support stakeholders investigate recurring issues and potential opportunities. Analytical signals are not treated as causal proof without further validation.

## Project Structure

```text
vireo_assessment/
├── data/
│   ├── agents.csv
│   ├── customers.csv
│   ├── orders.csv
│   ├── products.csv
│   └── tickets.csv
│
├── reports/
│   ├── 01_data_audit.md
│   ├── 02_business_analysis.md
│   ├── 03_repeat_analysis.md
│   ├── 05_business_goal.md
│   ├── audit_report.json
│   ├── business_analysis.json
│   ├── business_goal.json
│   ├── complaint_analysis.json
│   ├── complaint_examples.csv
│   ├── complaint_theme_metrics.csv
│   ├── complaint_themed_tickets.csv
│   ├── repeat_analysis.json
│   ├── repeat_candidates.csv
│   ├── repeat_validation_instructions.md
│   ├── repeat_validation_sample.csv
│   ├── repeat_validation_sample_completed.csv
│   ├── weekly_complaint_metrics.csv
│   └── weekly_support_digest.md
│
├── src/
│   ├── 01_data_audit.py
│   ├── 02_business_analysis.py
│   ├── 03_repeat_detection.py
│   ├── 04_validate_repeat_detection.py
│   ├── 05_business_goal.py
│   └── 06_complaint_analysis.py
│
├── app.py
├── support-policy.pdf
├── email-thread.txt
├── README.md
├── requirements.txt
└── .gitignore
```

## Analysis Pipeline

### Step 1 — Data Audit

```bash
python src/01_data_audit.py
```

Checks the supplied datasets, data quality, duplicates, missing values, and structural issues.

### Step 2 — Business Analysis

```bash
python src/02_business_analysis.py
```

Calculates support-contact costs, SLA breach exposure, transfer costs, refund exposure, and replacement-related metrics using the supplied operating policy.

### Step 3 — Repeat Contact Detection

```bash
python src/03_repeat_detection.py
```

Uses customer/ticket history and TF-IDF similarity to identify candidate repeat contacts.

The candidate rate is treated as a signal until manually validated.

### Step 4 — Repeat Candidate Validation

```bash
python src/04_validate_repeat_detection.py
```

Creates a validation sample so candidate repeat contacts can be manually labelled as:

- TRUE
- FALSE
- UNCLEAR

### Step 5 — Business Goal Analysis

```bash
python src/05_business_goal.py
```

Connects the observed operational signals to potential business opportunities without claiming that an intervention has already achieved a reduction.

### Step 6 — AI-Assisted Complaint Analysis

```bash
python src/06_complaint_analysis.py
```

Uses TF-IDF and KMeans clustering to discover recurring complaint themes and generate:

- Complaint theme metrics
- Themed tickets
- Example customer messages
- Weekly complaint metrics
- Weekly support digest
- Complaint analysis JSON

## Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The dashboard contains:

1. Executive Summary
2. Weekly Digest
3. Complaint Themes
4. Agent Leaderboard
5. Repeat Contacts
6. Evidence

## Important Analytical Notes

### Duplicate Tickets

The supplied export contains duplicate ticket IDs. The analysis deduplicates ticket records before calculating the analytical ticket population.

### CSAT

Blank CSAT values represent non-responses and should not be treated as zero when calculating averages.

### Repeat Contacts

Repeat-contact detection is a candidate-generation method. Candidate rates should not be presented as a validated KPI without manual validation.

### Complaint Themes

Complaint themes are machine-discovered using TF-IDF and clustering. Theme names are interpretations of cluster terms, so individual tickets should be reviewed before treating a theme as a definitive customer-intent category.

### Agent Metrics

Ticket volume is an operational view, not a universal productivity measure. Tier 2 warranty work has different case complexity and should not be compared with Tier 1 teams using ticket count alone.

### Causality

Historical associations between complaint themes and refunds, replacements, transfers, or CSAT do not by themselves establish causation.

## Source Documents

The project uses the supplied:

- `support-policy.pdf` — Vireo Audio Support Operating Policy v3.2
- `email-thread.txt` — communication between Vireo stakeholders and the delivery team
- CSV files in `data/`

No paid LLM/API calls are required for the complaint-theme analysis.

## Installation

Recommended Python environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Complete Analysis

Run the scripts in order:

```bash
python src/01_data_audit.py
python src/02_business_analysis.py
python src/03_repeat_detection.py
python src/04_validate_repeat_detection.py
python src/05_business_goal.py
python src/06_complaint_analysis.py
```

Then launch:

```bash
streamlit run app.py
```

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- KMeans clustering
- Streamlit
- Plotly
- Joblib

## Output

Generated analytical outputs are stored in:

```text
reports/
```

The Streamlit dashboard reads these generated reports to provide an interactive view of the analysis.
