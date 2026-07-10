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

> **MASS AI** is a machine-learning prototype for detecting electricity-theft signals and anomalies from smart-meter data. It is designed for utility decision-support workflows in Turkey and should be evaluated with documented partner data before operational claims are made.

---

## Real Data Validation (2026)

In May 2026, the project added an integration layer for compatible daily-column inputs and a controlled proxy evaluation.

The repository can map a compatible SGCC-style CSV into the current feature schema, but it does not include a downloaded public SGCC dataset or a completed real-SGCC benchmark.

**Important validation note:** the default committed benchmark uses an SGCC-style realistic proxy, not a downloaded public SGCC file. This is useful for measuring domain-shift risk, but it is not a final real-world claim. A real SGCC CSV or Turkish DSO pilot dataset is still required before production or field-performance claims.

**Benchmark Results (May 2026):**

| Dataset                    | AUC    | F1     |
|---------------------------|--------|--------|
| Synthetic (Turkey Urban)  | 0.999  | 0.97   |
| SGCC-style Realistic Proxy| 0.912  | 0.80   |
| **Measured Gap**          | **~0.087** | —   |

This gap is now measured and documented. Full tooling and automated reports are available.

See [the validation summary](docs/product/REAL_DATA_VALIDATION_SUMMARY.md) and run:
```bash
python scripts/benchmark_real_vs_synthetic.py
```

For stricter validation rules, see:

- [Real-data requirements](docs/product/REAL_DATA_REQUIREMENTS.md)
- [Model validation](docs/product/MODEL_VALIDATION.md)

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
| 📊 | **Data Adapter & Proxy Check** | SGCC-style input adapter and a generated proxy gap (~0.09 AUC) |
| 🗂️ | **Ops Center** | Basic case management |
| 🌐 | **Web Dashboard** | Streamlit UI (heavily cleaned in 2026) |

**Current Focus:** Preparing for university incubation programs with emphasis on real data transparency.

---

## Development

### Setup

```bash
pip install -r shared/requirements.txt
pip install -r requirements-dev.txt
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

**Docker:**
```bash
docker compose up --build
```

Then open `http://localhost:8501`.

### Tests

```bash
python -m pytest shared/tests/ -v
```

### Common Commands

```bash
make install
make test
make coverage
make benchmark
make dashboard
```

---

## Incubation Materials (2026)

Incubation-facing material is organized by purpose and keeps demonstrated evidence separate from plans and pricing hypotheses.

See the [documentation index](docs/README.md).

Operational readiness documents:

- [Pilot readiness and production path](docs/product/PRODUCTION_READINESS.md)
- [Pilot data request](docs/product/PILOT_DATA_REQUEST.md)

---

## Current Status

- Code quality significantly improved
- Real data gap measured and documented
- 42 tests passing
- Strong documentation for incubation
- Docker-based local deployment added

## Remaining Critical Gaps

- Public real SGCC or Turkish DSO pilot benchmark is still needed.
- `new_web/dashboard/app.py` remains large and should be split further.
- Authentication, structured logging, and staging hardening are still pending.
- Pilot traction is not proven until at least one partner reviews real or anonymized operational data.

The project is in a much cleaner and more honest state than earlier in 2026.

---

## License

MIT

---

*Last updated: July 2026*
