<div align="center">

# MASS AI Project

### Milli Akıllı Sayaç Sistemleri

**AI-Powered Electricity Theft Detection for Turkey's National Smart Meter Infrastructure**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EC4E20)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-000000?logo=vercel&logoColor=white)](https://mass-ai-project.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

<br/>

[Live Website](https://mass-ai-project.vercel.app/) · [Repository](https://github.com/Technet43/MASS-AI-Project)

</div>

<br/>

> **MASS AI** is a production-ready machine learning platform that detects electricity theft and consumption anomalies from smart meter data. Built for Turkey's MASS initiative (50 million smart meters by 2028), it targets regions where theft rates exceed **28%** — causing an estimated **₺10B+ in annual losses**.

---

## Real Data Validation (May 2026)

One of the most important improvements in 2026 was building a proper **real data validation layer**.

We created a full production-grade integration for public electricity theft datasets (starting with SGCC, the most widely used academic benchmark).

**Latest Benchmark Results:**

| Dataset                    | AUC     | F1     |
|---------------------------|---------|--------|
| Synthetic (Turkey Urban)  | 0.999   | 0.97   |
| SGCC-style Realistic Proxy| 0.912   | 0.80   |
| **Measured Gap**          | **~0.087** | -    |

This gap is now **measured and transparent**. We have complete tooling, automated reports, and clear documentation around it.

See:
- [docs/Real_Data_Validation_Summary.md](docs/Real_Data_Validation_Summary.md)
- `python scripts/benchmark_real_vs_synthetic.py`

---

## Current Architecture (2026)

**Single Source of Truth:** `shared/core/` (MassAIEngine + dashboard_adapters)

After a major modernization effort in 2026, the project now has a much cleaner structure:

- `shared/core/` → The only place containing real ML logic, synthetic data generation, model training, and real data adapters.
- `new_web/dashboard/` → Thin presentation layer (heavily cleaned and modernized).
- `docs/` → Strong incubation and technical documentation.
- `legacy/` → Old code moved here.

The dashboard was significantly refactored (reduced from ~2570 lines to ~2135 lines), with most duplicated logic removed and moved to clean adapters.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/Feature_Catalog.md](docs/Feature_Catalog.md) for details.

---

## Key Features

| | Feature | Description |
|---|---|---|
| 🤖 | **6 ML Models** | Isolation Forest, XGBoost, Random Forest, Gradient Boosting, LSTM Autoencoder + Stacking Ensemble |
| 🔍 | **8 Theft Patterns** | Realistic Turkish consumption theft behaviors (night zeroing, bypass, tampering, etc.) |
| 🧮 | **~40 Features** | Rich statistical, temporal, peer, and domain-specific features |
| 📊 | **Real Data Validation** | Full SGCC-style benchmark with measured ~0.09 AUC gap |
| 🗂️ | **Ops Center** | Case management with audit trail |
| 🌐 | **Modern Web Dashboard** | Clean Streamlit UI (major 2026 modernization) |
| ⚡ | **Synthetic + Real Data Engine** | Strong synthetic generator + production-ready real data support |

**Current Focus:** Incubation preparation (İTÜ Çekirdek & Yıldız Teknik) with emphasis on real data transparency and code quality.

---

## Development (Recommended Way)

### Setup

```bash
pip install -r shared/requirements.txt
```

### Running the Project

**Best experience:**
```bash
python apps/launcher/MASS_AI_LAUNCHER.py
```

**Run the Streamlit dashboard directly:**
```bash
streamlit run new_web/dashboard/app.py
```

### Running Tests

```bash
python -m pytest shared/tests/ -v
```

---

## Incubation Materials (2026)

During an extended deep work session in May 2026, the following professional materials were prepared:

- Full Pitch Deck Content
- Business Model & Revenue Scenarios
- Traction & Pilot Plan
- Real Data Validation Summary
- Incubation Readiness Checklist
- Updated One Pager

See [docs/Incubation_Materials_Index.md](docs/Incubation_Materials_Index.md) for the full list.

---

## Current Status

- **Code Quality:** Significantly improved (heavy dashboard modernization + architecture cleanup)
- **Real Data:** Full tooling + measured benchmark results
- **Tests:** 39 tests passing
- **Documentation:** Strong incubation package ready

The project is in a much cleaner and more professional state compared to early 2026.

---

## React + Vercel Website

**Live site:** https://mass-ai-project.vercel.app/

```bash
npm install
npm run dev
```

---

## License

MIT License

---

*Last major update: May 2026 (during extended deep work & incubation preparation session)*
