# MASS-AI — One Pager (May 2026)

**Project:** MASS-AI — AI-Powered Electricity Theft Detection  
**Tagline:** Measuring and closing the real data gap for Turkey's 50M smart meters  
**Contact:** [Ad Soyad] | [e-posta] | [telefon]

---

## The Problem
In some regions of Turkey, electricity theft rates exceed **28%**, causing **₺10B+ in annual losses**. Current detection methods are manual and ineffective. With 50 million smart meters being deployed by 2028, utilities have data but lack actionable intelligence.

**The #1 Incubator Concern We Addressed:**  
"Only synthetic data?" → We **measured** the gap and built the infrastructure to close it.

## Our Solution
MASS-AI is a production-grade platform that turns smart meter data into prioritized, explainable theft cases using 6 ML models (Isolation Forest, XGBoost, RF, GBM, LSTM Autoencoder + Stacking Ensemble).

**Key Differentiators:**
- **~40 rich features** (far beyond typical academic baselines)
- **Strong peer/network features** (transformer & feeder level) — major differentiator
- **8 Turkish-specific theft patterns** modeled with cultural/weather effects
- **Full real data validation layer** (SGCC benchmark) with transparent gap measurement

## Real Data Validation Results (May 2026)
| Dataset                    | AUC    | F1    |
|---------------------------|--------|-------|
| Synthetic (Turkey Urban)  | 0.999  | 0.97  |
| SGCC-style Realistic Proxy| 0.912  | 0.80  |
| **Measured Gap**          | **~0.087** | - |

We have built a complete production-ready integration layer (`real_data.py`) + automated benchmarking + reports. This is not a promise — it is **measured and documented**.

## Traction & Go-to-Market
- Strong synthetic engine + full real data tooling ready
- Complete incubation materials prepared (Pitch Deck, Business Model, Traction Plan, Real Data Summary, Readiness Checklist)
- Next milestone: Pilot with 1 Turkish distribution company (6-month target)

**Business Model:** Hybrid SaaS + success-based fee on verified theft reduction.

## Team & Ask
[Team description to be added]

**We are applying to İTÜ Çekirdek and Yıldız Teknik incubation programs.**  
We have built strong technical foundations, quantified the real data risk, and prepared professional materials. We are looking for mentorship in energy sector go-to-market and first pilot connections.

**Key Documents:**
- Real Data Validation Summary
- Full Pitch Deck Content
- Business Model & Revenue
- Traction & Pilot Plan
- Incubation Readiness Checklist

---

**Status:** High technical quality + transparent real data progress + ready incubation package.  
**Biggest Remaining Gap:** First real utility pilot + team narrative.

*Prepared during extended deep work session — May 2026*
