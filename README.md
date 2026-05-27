<div align="center">

# MASS AI Project

### Milli Akıllı Sayaç Sistemleri

**AI-Powered Electricity Theft Detection for Turkey's National Smart Meter Infrastructure**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EC4E20)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

<br/>

[Live Website](https://mass-ai-project.vercel.app/) · [Repository](https://github.com/Technet43/MASS-AI-Project)

</div>

<br/>

> **MASS AI** is a machine learning platform designed to detect electricity theft and anomalies from smart meter data. It was built with Turkey's national smart meter rollout (50 million meters by 2028) in mind, targeting regions where non-technical losses can exceed 28%.

---

## Real Data Validation (2026)

In May 2026, a major focus was adding proper support for **real public datasets**.

We built a complete integration layer for the SGCC electricity theft dataset (the most common academic benchmark).

**Benchmark Results (May 2026):**

| Dataset                    | AUC    | F1     |
|---------------------------|--------|--------|
| Synthetic (Turkey Urban)  | 0.999  | 0.97   |
| SGCC-style Realistic Proxy| 0.912  | 0.80   |
| **Measured Gap**          | **~0.087** | —   |

This gap is now measured and documented. Full tooling and automated reports are available.

See [docs/Real_Data_Validation_Summary.md](docs/Real_Data_Validation_Summary.md) and run:
```bash
python scripts/benchmark_real_vs_synthetic.py
```

---

## Current Architecture (2026)

After a major cleanup and modernization effort:

- **Single Source of Truth**: `shared/core/` (MassAIEngine + clean adapters)
- `new_web/dashboard/` is now a much thinner presentation layer
- Legacy code has been moved to `legacy/`
- The dashboard was heavily refactored (reduced by ~400+ lines, many deprecated functions removed)

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

---

## Key Features

| | Feature | Description |
|---|---|---|
| 🤖 | **5 Models + Stacking** | Isolation Forest, XGBoost, Random Forest, Gradient Boosting + Stacking Ensemble |
| 🔍 | **8 Theft Patterns** | Realistic Turkish consumption behaviors |
| 🧮 | **~40 Features** | Statistical, temporal, peer and domain features |
| 📊 | **Real Data Support** | SGCC-style benchmark with measured gap (~0.09 AUC) |
| 🗂️ | **Ops Center** | Basic case management |
| 🌐 | **Web Dashboard** | Streamlit UI (heavily cleaned in 2026) |

**Current Focus:** Preparing for university incubation programs with emphasis on real data transparency.

---

## Development

### Setup

```bash
pip install -r shared/requirements.txt
```

### Running

**Recommended:**
```bash
python apps/launcher/MASS_AI_LAUNCHER.py
```

**Dashboard only:**
```bash
streamlit run new_web/dashboard/app.py
```

### Tests

```bash
python -m pytest shared/tests/ -v
```

---

## Incubation Materials (2026)

During an extended work session in May 2026, the following materials were prepared:

- Full Pitch Deck content
- Business Model & Revenue scenarios
- Traction & Pilot Plan
- Real Data Validation Summary
- Incubation Readiness Checklist

See [docs/Incubation_Materials_Index.md](docs/Incubation_Materials_Index.md).

---

## Current Status

- Code quality significantly improved
- Real data gap measured and documented
- 39 tests passing
- Strong documentation for incubation

The project is in a much cleaner and more honest state than earlier in 2026.

---

## License

MIT

---

*Last updated: May 2026*
