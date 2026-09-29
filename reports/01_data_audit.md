# Vireo Audio — Data Audit Report

## 1. Dataset Overview

| Dataset | Rows |
|---|---:|
| Tickets | 12,528 |
| Agents | 44 |
| Customers | 9,500 |
| Orders | 15,000 |
| Products | 14 |

## 2. Ticket Coverage

- Raw ticket rows: **12,528**
- Unique ticket IDs: **11,875**
- Duplicate ticket IDs: **653**
- Rows involved in duplicate IDs: **1,306**
- Earliest ticket: **2025-01-01 09:48:00**
- Latest ticket: **2026-06-30 23:29:00**

## 3. Referential Integrity

| Relationship | Unknown IDs |
|---|---:|
| Customer | 0 |
| Product | 0 |
| Agent | 0 |
| Order | 0 |

## 4. Important Data Decisions

1. Ticket-level analysis will use unique `ticket_id` records.
2. Duplicate ticket IDs will be investigated because the supplied context describes legacy migration and re-imports.
3. Legacy CSAT value `0` will be treated as no response.
4. Missing CSAT responses will be excluded from CSAT averages.
5. Referential integrity will be checked before joining datasets.

## 5. Initial Conclusion

The ticket export contains duplicate ticket IDs, so raw row counts should not be treated as ticket counts without investigation.

The next analysis stage will examine the cleaned ticket population for business-impact opportunities including repeat contacts, SLA breaches, transfers, refunds, replacements and customer-satisfaction patterns.
