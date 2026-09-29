# Vireo Audio — Repeat Contact Analysis

## Purpose

This analysis identifies **potential repeat contacts** where the
same customer contacts support again within a 30-day window.

The detector is a candidate-generation tool, not ground truth.
Its accuracy must be manually validated before the repeat-contact
rate is used as a final business KPI.

---

## Population

- Unique tickets analyzed: **11,875**

---

## Candidate Results

| Metric | Result |
|---|---:|
| Repeat candidates | 363 |
| Repeat candidate rate | 3.06% |
| Strong repeat candidates | 356 |
| Strong candidate rate | 3.00% |

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

**0.35**

---

## Customer Experience Comparison

| Metric | Repeat Candidates | Other Tickets |
|---|---:|---:|
| Average CSAT | 2.72 | 3.33 |
| Tickets with transfer | 12.08% | 8.66% |

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
