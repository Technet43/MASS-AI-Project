from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# TODO: centralised logging — writes to logs/mass_ai.log
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "mass_ai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_logger = logging.getLogger("mass_ai_engine")

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    _logger.warning("joblib not installed — model persistence disabled")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mass_ai_domain import (
    COLUMN_ALIASES,
    CRITICAL_RISK_LABELS,
    RISK_LABELS,
    fmt_currency,
    fmt_percent,
    normalize_column_key,
    safe_text,
)

DEFAULT_SYNTHETIC_PRESET = "Turkey Urban"
THEFT_PATTERNS = [
    "constant_reduction",
    "night_zeroing",
    "random_zeros",
    "gradual_decrease",
    "peak_clipping",
    "weekend_masking",
    "intermittent_bypass",
    "tamper_spikes",
]
SYNTHETIC_PRESETS: dict[str, dict[str, Any]] = {
    "Turkey Urban": {
        "description": "Dense metro-heavy utility mix with residential towers, mixed-use feeders, and strong night tamper behavior.",
        "profile_weights": {"residential": 0.50, "commercial": 0.22, "industrial": 0.08, "mixed_use": 0.20},
        "region_weights": {"coastal": 0.16, "metro": 0.58, "plateau": 0.16, "rural": 0.10},
        "contract_weights": {"standard": 0.42, "time_of_use": 0.43, "large_scale": 0.15},
        "meter_health_weights": {"healthy": 0.67, "aging": 0.22, "tampered": 0.11},
        "theft_rate": 0.15,
        "theft_weights": {"constant_reduction": 0.10, "night_zeroing": 0.23, "random_zeros": 0.14, "gradual_decrease": 0.08, "peak_clipping": 0.09, "weekend_masking": 0.08, "intermittent_bypass": 0.14, "tamper_spikes": 0.14},
        "base_multiplier": {"residential": 1.05, "commercial": 1.03, "industrial": 0.94, "mixed_use": 1.07},
        "outage_bias": {"coastal": 0.8, "metro": 0.75, "plateau": 1.0, "rural": 1.1},
        "solar_cap": 0.24,
        "tamper_boost": 1.15,
    },
    "Industrial Theft Sweep": {
        "description": "Heavy industrial and commercial feeders with larger contract demand, concentrated bypass activity, and high exposure losses.",
        "profile_weights": {"residential": 0.14, "commercial": 0.29, "industrial": 0.39, "mixed_use": 0.18},
        "region_weights": {"coastal": 0.12, "metro": 0.34, "plateau": 0.29, "rural": 0.25},
        "contract_weights": {"standard": 0.20, "time_of_use": 0.30, "large_scale": 0.50},
        "meter_health_weights": {"healthy": 0.59, "aging": 0.20, "tampered": 0.21},
        "theft_rate": 0.18,
        "theft_weights": {"constant_reduction": 0.16, "night_zeroing": 0.06, "random_zeros": 0.07, "gradual_decrease": 0.10, "peak_clipping": 0.21, "weekend_masking": 0.07, "intermittent_bypass": 0.19, "tamper_spikes": 0.14},
        "base_multiplier": {"residential": 0.82, "commercial": 1.08, "industrial": 1.28, "mixed_use": 1.11},
        "outage_bias": {"coastal": 0.9, "metro": 0.9, "plateau": 1.0, "rural": 1.2},
        "solar_cap": 0.10,
        "tamper_boost": 1.35,
    },
    "Mixed Retail Anomalies": {
        "description": "Retail and mixed-use corridors with volatile weekend masking, intermittent zeros, and time-of-use shifts around closing hours.",
        "profile_weights": {"residential": 0.20, "commercial": 0.36, "industrial": 0.08, "mixed_use": 0.36},
        "region_weights": {"coastal": 0.24, "metro": 0.46, "plateau": 0.18, "rural": 0.12},
        "contract_weights": {"standard": 0.28, "time_of_use": 0.56, "large_scale": 0.16},
        "meter_health_weights": {"healthy": 0.71, "aging": 0.18, "tampered": 0.11},
        "theft_rate": 0.13,
        "theft_weights": {"constant_reduction": 0.11, "night_zeroing": 0.10, "random_zeros": 0.18, "gradual_decrease": 0.06, "peak_clipping": 0.11, "weekend_masking": 0.24, "intermittent_bypass": 0.12, "tamper_spikes": 0.08},
        "base_multiplier": {"residential": 0.94, "commercial": 1.12, "industrial": 0.88, "mixed_use": 1.15},
        "outage_bias": {"coastal": 0.9, "metro": 0.85, "plateau": 1.0, "rural": 1.05},
        "solar_cap": 0.18,
        "tamper_boost": 1.05,
    },
    "Rural Meter Drift": {
        "description": "Rural feeders with aging meters, higher outage pressure, slower consumption drift, and low-visibility tamper signals.",
        "profile_weights": {"residential": 0.61, "commercial": 0.15, "industrial": 0.07, "mixed_use": 0.17},
        "region_weights": {"coastal": 0.08, "metro": 0.10, "plateau": 0.28, "rural": 0.54},
        "contract_weights": {"standard": 0.62, "time_of_use": 0.20, "large_scale": 0.18},
        "meter_health_weights": {"healthy": 0.52, "aging": 0.34, "tampered": 0.14},
        "theft_rate": 0.11,
        "theft_weights": {"constant_reduction": 0.08, "night_zeroing": 0.17, "random_zeros": 0.16, "gradual_decrease": 0.22, "peak_clipping": 0.08, "weekend_masking": 0.07, "intermittent_bypass": 0.10, "tamper_spikes": 0.12},
        "base_multiplier": {"residential": 0.96, "commercial": 0.92, "industrial": 0.90, "mixed_use": 0.95},
        "outage_bias": {"coastal": 1.0, "metro": 0.85, "plateau": 1.15, "rural": 1.45},
        "solar_cap": 0.20,
        "tamper_boost": 1.10,
    },
}


def _normalized_weights(mapping: dict[str, float]) -> tuple[list[str], list[float]]:
    labels = list(mapping.keys())
    values = np.array(list(mapping.values()), dtype=float)
    values = values / values.sum()
    return labels, values.tolist()


def _humanize_pattern(value: Any) -> str:
    return safe_text(value).replace("_", " ")


# TODO: ModelRegistry — joblib-backed versioned model persistence
class ModelRegistry:
    """Persist and load versioned ML models with joblib.

    Version key is derived from a SHA-1 hash of the feature column list so the
    same model file is never overwritten by an incompatible schema.
    """

    def __init__(self, registry_dir: str | Path | None = None):
        if registry_dir is None:
            registry_dir = Path(__file__).resolve().parent.parent / "model_registry"
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, dict[str, Any]] = {}

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _schema_hash(feature_cols: list[str]) -> str:
        return hashlib.sha1("|".join(sorted(feature_cols)).encode()).hexdigest()[:8]

    def _model_path(self, model_name: str, schema_hash: str) -> Path:
        safe = model_name.replace(" ", "_").lower()
        return self.registry_dir / f"{safe}_{schema_hash}.joblib"

    # ── public API ────────────────────────────────────────────────────────────
    def save(
        self,
        model_name: str,
        model_obj: Any,
        feature_cols: list[str],
        metrics: dict[str, float],
        scaler: Any = None,
    ) -> Path:
        """Serialise model + scaler to disk and record in the manifest."""
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required for model persistence")
        schema_hash = self._schema_hash(feature_cols)
        path = self._model_path(model_name, schema_hash)
        payload = {
            "model": model_obj,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        joblib.dump(payload, path, compress=3)
        self._manifest[model_name] = {
            "path": str(path),
            "schema_hash": schema_hash,
            "metrics": metrics,
            "saved_at": payload["saved_at"],
        }
        _logger.info("ModelRegistry: saved '%s' → %s (AUC=%.4f)", model_name, path.name, metrics.get("auc", 0))
        return path

    def load(self, model_name: str, feature_cols: list[str]) -> dict[str, Any] | None:
        """Load a previously saved model. Returns None if not found."""
        if not _JOBLIB_AVAILABLE:
            return None
        schema_hash = self._schema_hash(feature_cols)
        path = self._model_path(model_name, schema_hash)
        if not path.exists():
            _logger.debug("ModelRegistry: '%s' not found on disk (%s)", model_name, path.name)
            return None
        try:
            payload = joblib.load(path)
            _logger.info("ModelRegistry: loaded '%s' from %s", model_name, path.name)
            return payload
        except Exception as exc:
            _logger.error("ModelRegistry: failed to load '%s': %s", model_name, exc)
            return None

    def best_model(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """Return (model_name, manifest_entry) for the model with the highest AUC."""
        if not self._manifest:
            return None, None
        best_name = max(self._manifest, key=lambda n: self._manifest[n]["metrics"].get("auc", 0))
        return best_name, self._manifest[best_name]

    def list_versions(self) -> list[dict[str, Any]]:
        return [
            {"model_name": name, **entry}
            for name, entry in sorted(
                self._manifest.items(),
                key=lambda item: item[1]["metrics"].get("auc", 0),
                reverse=True,
            )
        ]


class MassAIEngine:
    def __init__(self):
        self.df_raw = None
        self.df_features = None
        self.df_scored = None
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.feature_cols: list[str] = []
        self.scaler = None
        self.log_lines: list[str] = []
        self.last_source = "No dataset loaded yet"
        self.last_run_at = None
        self.last_preset_name: str | None = None
        self.last_preset_summary: str | None = None
        self.last_explainability_summary: str | None = None
        # TODO: attach ModelRegistry for joblib-backed versioned persistence
        self.registry = ModelRegistry()

    def reset_state(self):
        self.df_raw = None
        self.df_features = None
        self.df_scored = None
        self.models = {}
        self.results = {}
        self.feature_cols = []
        self.scaler = None
        self.last_explainability_summary = None

    def log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{stamp}] {message}")

    def synthetic_preset_names(self) -> list[str]:
        return list(SYNTHETIC_PRESETS.keys())

    def _resolve_synthetic_preset(self, preset_name: str | None) -> tuple[str, dict[str, Any]]:
        normalized_name = safe_text(preset_name)
        if normalized_name in SYNTHETIC_PRESETS:
            return normalized_name, SYNTHETIC_PRESETS[normalized_name]
        return DEFAULT_SYNTHETIC_PRESET, SYNTHETIC_PRESETS[DEFAULT_SYNTHETIC_PRESET]

    def best_model_name(self):
        if self.results and any(self.results[name].get("auc", 0) > 0 for name in self.results):
            return max(self.results, key=lambda name: self.results[name].get("auc", 0))
        if self.results:
            return "Isolation Forest"
        return "No model yet"

    def generate_synthetic(self, n_customers=2000, n_days=180, callback=None, preset_name: str | None = None):
        preset_name, preset = self._resolve_synthetic_preset(preset_name)
        self.reset_state()
        self.last_preset_name = preset_name
        self.last_preset_summary = preset["description"]
        self.last_source = f"Synthetic dataset | {preset_name} | {n_customers} customers"
        self.log(f"Generating synthetic data: {n_customers} customers, {n_days} days")
        self.log(f"Synthetic preset: {preset_name} - {preset['description']}")
        if callback:
            callback(5, f"Creating {preset_name} profiles")

        profile_names, profile_probs = _normalized_weights(preset["profile_weights"])
        region_names, region_probs = _normalized_weights(preset["region_weights"])
        contract_names, contract_probs = _normalized_weights(preset["contract_weights"])
        meter_names, meter_probs = _normalized_weights(preset["meter_health_weights"])
        theft_names, theft_probs = _normalized_weights(preset["theft_weights"])

        profiles = np.random.choice(profile_names, n_customers, p=profile_probs)
        regions = np.random.choice(region_names, n_customers, p=region_probs)
        contract_types = np.random.choice(contract_names, n_customers, p=contract_probs)
        meter_health = np.random.choice(meter_names, n_customers, p=meter_probs)
        tariff_groups = np.array(["T1" if p == "residential" else "T2" if p in {"commercial", "mixed_use"} else "T3" for p in profiles])
        n_transformers = max(n_customers // 40, 2)
        n_feeders = max(n_transformers // 5, 1)
        transformer_ids = np.array([f"TR-{i % n_transformers + 1:03d}" for i in range(n_customers)])
        np.random.shuffle(transformer_ids)
        feeder_ids = np.array([f"FD-{(int(tid.split('-')[1]) - 1) // max(n_transformers // n_feeders, 1) + 1:02d}" for tid in transformer_ids])
        
        # Geolocation for Turkey bounds
        lats = np.random.uniform(36.8, 41.8, n_customers)
        lons = np.random.uniform(26.5, 44.0, n_customers)

        labels = np.zeros(n_customers, dtype=int)
        theft_count = max(int(n_customers * float(preset["theft_rate"])), 1)
        theft_indices = np.random.choice(n_customers, theft_count, replace=False)
        labels[theft_indices] = 1

        theft_types = np.full(n_customers, "none", dtype=object)
        theft_types[theft_indices] = np.random.choice(theft_names, size=len(theft_indices), p=theft_probs)

        rows = []
        hours = np.tile(np.arange(24), n_days)
        day_index = np.repeat(np.arange(n_days), 24)
        day_of_week = np.repeat(np.arange(n_days) % 7, 24)
        weekend_mask = day_of_week >= 5
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * day_index / 365)
        monthly_cycle = 1 + 0.06 * np.cos(2 * np.pi * day_index / 30)
        weather_wave = 18 + 10 * np.sin(2 * np.pi * day_index / 365) + 3 * np.sin(2 * np.pi * day_index / 14)
        region_weather_adjustment = {"coastal": 1.02, "metro": 1.0, "plateau": 1.08, "rural": 0.97}

        for customer_idx in range(n_customers):
            if callback and customer_idx % 200 == 0:
                progress = 15 + int((customer_idx / n_customers) * 45)
                callback(progress, f"Processing customer {customer_idx}/{n_customers}")

            profile = profiles[customer_idx]
            region = regions[customer_idx]
            contract_type = contract_types[customer_idx]
            health = meter_health[customer_idx]
            profile_multiplier = float(preset["base_multiplier"].get(profile, 1.0))
            base = {
                "residential": 1.4 + np.random.random() * 2.4,
                "commercial": 3.2 + np.random.random() * 5.8,
                "industrial": 11 + np.random.random() * 22,
                "mixed_use": 4.5 + np.random.random() * 6.5,
            }[profile]
            base *= region_weather_adjustment[region] * profile_multiplier

            weekend_factor = 0.86 if profile in {"commercial", "industrial"} else 1.12
            solar_cap = float(preset.get("solar_cap", 0.24))
            solar_factor = np.random.uniform(0.0, solar_cap if profile != "industrial" else solar_cap * 0.35)
            meter_age_years = np.random.randint(1, 16)
            contract_demand_kw = base * np.random.uniform(9, 16)
            outage_lambda = (1.5 if region == "rural" else 0.8) * float(preset["outage_bias"].get(region, 1.0))
            outage_count = np.random.poisson(outage_lambda)
            outage_days = np.random.choice(n_days, size=min(outage_count, max(n_days // 12, 1)), replace=False) if outage_count else np.array([], dtype=int)
            outage_profile = np.ones(len(hours))
            for outage_day in outage_days:
                start_hour = np.random.randint(1, 20)
                duration = np.random.randint(2, 6)
                window = (day_index == outage_day) & (hours >= start_hour) & (hours < start_hour + duration)
                outage_profile[window] *= np.random.uniform(0.02, 0.15)

            if profile == "residential":
                hourly_pattern = np.where(
                    (hours >= 7) & (hours <= 9),
                    1.8,
                    np.where((hours >= 18) & (hours <= 23), 2.2, 0.3),
                )
            elif profile == "commercial":
                hourly_pattern = np.where((hours >= 9) & (hours <= 18), 1.5, 0.2)
            elif profile == "mixed_use":
                hourly_pattern = np.where(
                    (hours >= 8) & (hours <= 11),
                    1.2,
                    np.where((hours >= 17) & (hours <= 22), 1.7, 0.4),
                )
            else:
                hourly_pattern = 0.8 + np.random.random(len(hours)) * 0.4

            weekend_profile = np.where(weekend_mask, weekend_factor, 1.0)
            daytime_solar_relief = np.where((hours >= 11) & (hours <= 15), 1 - solar_factor, 1.0)
            heat_sensitivity = 1 + ((weather_wave - weather_wave.mean()) / max(weather_wave.std(), 1e-6)) * np.random.uniform(0.01, 0.06)
            contract_shape = 1.04 if contract_type == "time_of_use" and profile in {"commercial", "mixed_use"} else 1.0
            noise = np.random.normal(0, base * (0.12 if health == "healthy" else 0.18), len(hours))

            consumption = base * hourly_pattern * seasonal * monthly_cycle * weekend_profile * daytime_solar_relief * heat_sensitivity * contract_shape
            consumption = consumption * outage_profile + noise
            consumption = np.maximum(consumption, 0)

            theft_type = theft_types[customer_idx]
            if theft_type == "constant_reduction":
                consumption *= np.random.uniform(0.3, 0.5)
            elif theft_type == "night_zeroing":
                night_mask = hours < 6
                consumption[night_mask] = np.random.uniform(0, 0.02, night_mask.sum())
            elif theft_type == "random_zeros":
                zero_mask = np.random.random(len(consumption)) < 0.3
                consumption[zero_mask] = 0
            elif theft_type == "gradual_decrease":
                consumption *= np.linspace(1.0, 0.4, len(consumption))
            elif theft_type == "peak_clipping":
                threshold = np.percentile(consumption, 60)
                consumption = np.minimum(consumption, threshold)
            elif theft_type == "weekend_masking":
                consumption[weekend_mask] *= np.random.uniform(0.35, 0.6)
            elif theft_type == "intermittent_bypass":
                bypass_days = np.random.random(n_days) < 0.18
                bypass_mask = np.repeat(bypass_days, 24)
                consumption[bypass_mask] *= np.random.uniform(0.08, 0.35)
            elif theft_type == "tamper_spikes":
                spike_mask = np.random.random(len(consumption)) < 0.08
                consumption[spike_mask] *= np.random.uniform(0.0, 0.1)

            if health == "aging":
                drift = np.linspace(1.0, np.random.uniform(0.92, 0.98), len(consumption))
                consumption *= drift
            elif health == "tampered" and theft_type == "none":
                tamper_noise = np.random.random(len(consumption)) < 0.06 * float(preset.get("tamper_boost", 1.0))
                consumption[tamper_noise] *= np.random.uniform(0.15, 0.55)

            daily_totals = consumption.reshape(n_days, 24).sum(axis=1)
            weekday_daily = daily_totals[np.arange(n_days) % 7 < 5]
            weekend_daily = daily_totals[np.arange(n_days) % 7 >= 5]
            diff = np.abs(np.diff(consumption))
            diff_mean = np.mean(diff)
            peak_window = consumption[(hours >= 18) & (hours <= 22)]
            offpeak_window = consumption[(hours >= 1) & (hours <= 5)]
            baseload = np.percentile(consumption, 10)
            morning_window = consumption[(hours >= 6) & (hours <= 10)]
            noon_window = consumption[(hours >= 11) & (hours <= 15)]
            event_spike_ratio = np.mean(consumption > np.percentile(consumption, 95))
            load_factor = np.mean(consumption) / (np.max(consumption) + 1e-8)
            bill_proxy = daily_totals * np.where(np.arange(n_days) % 7 >= 5, 0.92, 1.0)
            rolling_std = pd.Series(daily_totals).rolling(7, min_periods=3).std().fillna(0)
            anomaly_burst_ratio = np.mean(rolling_std > rolling_std.mean() + rolling_std.std())
            temperature_correlation = float(np.corrcoef(daily_totals, weather_wave.reshape(n_days, 24).mean(axis=1))[0, 1]) if n_days > 2 else 0.0
            if np.isnan(temperature_correlation):
                temperature_correlation = 0.0

            is_thief = labels[customer_idx] == 1
            base_tamper = np.random.poisson(0.8 * float(preset.get("tamper_boost", 1.0))) if not is_thief else np.random.poisson(np.random.uniform(3, 8) * float(preset.get("tamper_boost", 1.0)))
            tamper_event_count = int(base_tamper)
            days_since_last_tamper = int(np.random.randint(1, max(n_days, 2))) if tamper_event_count > 0 else n_days
            tamper_density = round(tamper_event_count / max(n_days / 30, 1), 3)
            premise_density = "dense" if region == "metro" else "clustered" if region in {"coastal", "plateau"} else "dispersed"

            rows.append(
                {
                    "customer_id": customer_idx,
                    "profile": profile,
                    "region": region,
                    "contract_type": contract_type,
                    "meter_health": health,
                    "premise_density": premise_density,
                    "synthetic_preset": preset_name,
                    "transformer_id": transformer_ids[customer_idx],
                    "feeder_id": feeder_ids[customer_idx],
                    "latitude": float(lats[customer_idx]),
                    "longitude": float(lons[customer_idx]),
                    "tariff_group": tariff_groups[customer_idx],
                    "label": labels[customer_idx],
                    "theft_type": theft_type,
                    "mean_consumption": np.mean(consumption),
                    "std_consumption": np.std(consumption),
                    "min_consumption": np.min(consumption),
                    "max_consumption": np.max(consumption),
                    "median_consumption": np.median(consumption),
                    "skewness": float(pd.Series(consumption).skew()),
                    "kurtosis": float(pd.Series(consumption).kurtosis()),
                    "mean_daily_total": np.mean(daily_totals),
                    "std_daily_total": np.std(daily_totals),
                    "cv_daily": np.std(daily_totals) / (np.mean(daily_totals) + 1e-8),
                    "night_day_ratio": np.mean(consumption[hours < 6]) / (np.mean(consumption[(hours >= 9) & (hours <= 18)]) + 1e-8),
                    "zero_measurement_pct": np.mean(consumption < 0.01),
                    "sudden_change_ratio": np.mean(diff > 3 * diff_mean) if diff_mean > 0 else 0,
                    "trend_slope": np.polyfit(np.arange(len(daily_totals)), daily_totals, 1)[0],
                    "peak_hour": np.argmax(np.bincount(np.argmax(consumption.reshape(-1, 24), axis=1), minlength=24)),
                    "iqr": np.percentile(consumption, 75) - np.percentile(consumption, 25),
                    "weekend_weekday_ratio": np.mean(weekend_daily) / (np.mean(weekday_daily) + 1e-8) if len(weekday_daily) else 0,
                    "peak_offpeak_ratio": np.mean(peak_window) / (np.mean(offpeak_window) + 1e-8),
                    "morning_noon_ratio": np.mean(morning_window) / (np.mean(noon_window) + 1e-8),
                    "baseload_ratio": baseload / (np.mean(consumption) + 1e-8),
                    "load_factor": load_factor,
                    "event_spike_ratio": event_spike_ratio,
                    "billing_volatility": float(np.std(bill_proxy)),
                    "rolling_weekly_volatility": float(np.mean(rolling_std)),
                    "anomaly_burst_ratio": anomaly_burst_ratio,
                    "temperature_sensitivity": temperature_correlation,
                    "solar_relief_factor": solar_factor,
                    "meter_age_years": meter_age_years,
                    "contract_demand_kw": contract_demand_kw,
                    "outage_event_count": int(outage_count),
                    "tamper_event_count": tamper_event_count,
                    "days_since_last_tamper": days_since_last_tamper,
                    "tamper_density": tamper_density,
                }
            )

        self.df_features = pd.DataFrame(rows)
        trafo_totals = self.df_features.groupby("transformer_id")["mean_consumption"].transform("sum")
        technical_loss_pct = np.random.uniform(0.02, 0.05, len(self.df_features))
        trafo_supply = trafo_totals * (1 + technical_loss_pct)
        self.df_features["transformer_loss_pct"] = np.round((trafo_supply - trafo_totals) / (trafo_supply + 1e-8) * 100, 2)
        self.df_features["customer_share_of_loss"] = np.round(self.df_features["mean_consumption"] / (trafo_totals + 1e-8) * 100, 2)
        self.df_features["transformer_peer_count"] = self.df_features.groupby("transformer_id")["customer_id"].transform("count")

        peer_key = self.df_features["transformer_id"] + "_" + self.df_features["profile"]
        peer_mean = peer_key.map(self.df_features.groupby(peer_key)["mean_consumption"].mean())
        peer_zero_mean = peer_key.map(self.df_features.groupby(peer_key)["zero_measurement_pct"].mean())
        self.df_features["peer_consumption_ratio"] = np.round(self.df_features["mean_consumption"] / (peer_mean + 1e-8), 3)
        self.df_features["peer_zero_pct_deviation"] = np.round(self.df_features["zero_measurement_pct"] - peer_zero_mean, 4)
        self.df_features["peer_rank_in_transformer"] = self.df_features.groupby("transformer_id")["mean_consumption"].rank(ascending=True, method="min").astype(int)

        month = np.random.randint(1, 13, len(self.df_features))
        self.df_features["is_summer_peak"] = np.isin(month, [6, 7, 8]).astype(int)
        self.df_features["is_ramadan_period"] = np.isin(month, [3, 4]).astype(int)
        self.df_features["is_bayram_week"] = np.isin(month, [4, 6]).astype(int)
        seasonal_risk = (self.df_features["is_summer_peak"] & (self.df_features["profile"] == "residential")) | (
            self.df_features["is_ramadan_period"] & (self.df_features["profile"].isin(["commercial", "mixed_use"]))
        )
        self.df_features["seasonal_anomaly_flag"] = seasonal_risk.astype(int)

        self.log(f"Dataset ready: {len(self.df_features)} customers, {len(self.df_features.columns)} columns")
        self.log(f"Labeled fraud records: {(self.df_features['label'] == 1).sum()}")
        self.log(f"Transformers: {self.df_features['transformer_id'].nunique()}, Feeders: {self.df_features['feeder_id'].nunique()}")
        return self.df_features

    def _read_dataset(self, path):
        lower_path = str(path).lower()
        if lower_path.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(path)
            except ImportError as exc:
                raise ValueError("Excel support requires openpyxl. Install it with: python -m pip install openpyxl") from exc
        encodings = ["utf-8-sig", "utf-8", "cp1254", "latin1"]
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error:
            raise last_error
        return pd.read_csv(path)

    def _canonicalize_columns(self, frame):
        renamed = {}
        for column in frame.columns:
            key = normalize_column_key(column)
            canonical = None
            for target, aliases in COLUMN_ALIASES.items():
                if key in aliases:
                    canonical = target
                    break
            renamed[column] = canonical or key or str(column)
        frame = frame.rename(columns=renamed)
        unnamed = [col for col in frame.columns if str(col).startswith("unnamed")]
        if unnamed:
            frame = frame.drop(columns=unnamed)
        return frame

    def schema_summary(self) -> dict[str, Any]:
        if self.df_features is None:
            return {"raw_columns": [], "normalized_columns": [], "numeric_columns": [], "feature_columns": []}
        numeric_columns = self.df_features.select_dtypes(include=["number"]).columns.tolist()
        return {
            "raw_columns": list(self.df_raw.columns) if self.df_raw is not None else [],
            "normalized_columns": list(self.df_features.columns),
            "numeric_columns": numeric_columns,
            "feature_columns": list(self.feature_cols),
        }

    def load_dataset(self, path, callback=None):
        self.reset_state()
        self.last_preset_name = None
        self.last_preset_summary = "Imported dataset with flexible schema normalization."
        self.last_source = os.path.basename(path)
        self.log(f"Loading dataset: {os.path.basename(path)}")
        if callback:
            callback(10, "Reading file")

        self.df_raw = self._read_dataset(path)
        if self.df_raw.empty:
            raise ValueError("The selected file appears to be empty.")

        self.df_features = self._canonicalize_columns(self.df_raw.copy())
        if "label" not in self.df_features.columns:
            self.df_features["label"] = 0
        if "customer_id" not in self.df_features.columns:
            self.df_features["customer_id"] = np.arange(len(self.df_features))
        if "profile" not in self.df_features.columns:
            self.df_features["profile"] = "unknown"
        if "theft_type" not in self.df_features.columns:
            self.df_features["theft_type"] = "unknown"

        for col in self.df_features.columns:
            if col not in {"profile", "theft_type", "region", "contract_type", "meter_health", "premise_density", "synthetic_preset"}:
                try:
                    self.df_features[col] = pd.to_numeric(self.df_features[col])
                except (ValueError, TypeError):
                    pass

        self.log(f"Dataset ready: {self.df_features.shape[0]} rows, {self.df_features.shape[1]} columns")
        self.log(f"Schema summary: {len(self.schema_summary()['numeric_columns'])} numeric columns detected after normalization")
        self.log("Schema normalization applied to support region-specific column order and naming differences")
        return self.df_features

    def load_csv(self, path, callback=None):
        return self.load_dataset(path, callback=callback)

    def train_models(self, callback=None):
        if self.df_features is None or self.df_features.empty:
            raise ValueError("A dataset is required before training can start.")

        self.log("Model training started")
        if callback:
            callback(64, "Preparing model features")

        meta_cols = ["customer_id", "profile", "label", "theft_type"]
        numeric_cols = self.df_features.select_dtypes(include=["number"]).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in meta_cols]
        if not self.feature_cols:
            available = ", ".join(self.df_features.columns[:12])
            raise ValueError(f"No numeric features were found after schema normalization. Available columns: {available}")

        X = self.df_features[self.feature_cols].fillna(0).values
        y = self.df_features["label"].fillna(0).astype(int).values

        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        supervised_ok = len(np.unique(y)) > 1 and (y == 0).sum() >= 2 and (y == 1).sum() >= 2
        if supervised_ok:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled[: min(100, len(X_scaled))], y, y[: min(100, len(y))]
            self.log("Not enough labels were found. Falling back to the unsupervised scoring path.")

        negative = max((y_train == 0).sum(), 1)
        positive = max((y_train == 1).sum(), 1)

        if callback:
            callback(72, "Training Isolation Forest")
        iso = IsolationForest(n_estimators=220, contamination=max(y.mean(), 0.05), random_state=42)
        iso.fit(X_train)
        self.models["Isolation Forest"] = iso
        iso_scores_full = -iso.score_samples(X_scaled)
        if supervised_ok:
            iso_scores = -iso.score_samples(X_test)
            iso_pred = (iso.predict(X_test) == -1).astype(int)
            self.results["Isolation Forest"] = {
                "auc": roc_auc_score(y_test, iso_scores),
                "f1": f1_score(y_test, iso_pred),
                "type": "Unsupervised",
            }
        else:
            self.results["Isolation Forest"] = {"auc": 0.0, "f1": 0.0, "type": "Fallback"}

        if supervised_ok:
            if callback:
                callback(78, "Training XGBoost")
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=negative / positive,
                subsample=0.82,
                colsample_bytree=0.82,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            xgb.fit(X_train, y_train)
            self.models["XGBoost"] = xgb
            xgb_prob = xgb.predict_proba(X_test)[:, 1]
            self.results["XGBoost"] = {"auc": roc_auc_score(y_test, xgb_prob), "f1": f1_score(y_test, xgb.predict(X_test)), "type": "Supervised"}

            if callback:
                callback(84, "Training Random Forest")
            rf = RandomForestClassifier(n_estimators=240, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            self.models["Random Forest"] = rf
            rf_prob = rf.predict_proba(X_test)[:, 1]
            self.results["Random Forest"] = {
                "auc": roc_auc_score(y_test, rf_prob),
                "f1": f1_score(y_test, rf.predict(X_test)),
                "type": "Supervised",
                "importances": dict(zip(self.feature_cols, rf.feature_importances_)),
            }

            if callback:
                callback(89, "Training Gradient Boosting")
            gb = GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, subsample=0.82, random_state=42)
            gb.fit(X_train, y_train)
            self.models["Gradient Boosting"] = gb
            gb_prob = gb.predict_proba(X_test)[:, 1]
            self.results["Gradient Boosting"] = {"auc": roc_auc_score(y_test, gb_prob), "f1": f1_score(y_test, gb.predict(X_test)), "type": "Supervised"}

            if callback:
                callback(93, "Training Stacking Ensemble")
            stack = StackingClassifier(
                estimators=[
                    ("xgb", XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, scale_pos_weight=negative / positive, eval_metric="logloss", random_state=42, verbosity=0)),
                    ("rf", RandomForestClassifier(n_estimators=150, max_depth=7, class_weight="balanced", random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)),
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                stack_method="predict_proba",
                n_jobs=-1,
            )
            stack.fit(X_train, y_train)
            self.models["Stacking Ensemble"] = stack
            stack_prob = stack.predict_proba(X_test)[:, 1]
            self.results["Stacking Ensemble"] = {"auc": roc_auc_score(y_test, stack_prob), "f1": f1_score(y_test, stack.predict(X_test)), "type": "Meta Learning"}
        else:
            self.log("Supervised models were skipped because there were not enough class labels.")

        self.log(f"Training complete: {', '.join(self.models.keys())}")
        self.log(f"Isolation baseline scores prepared: min={iso_scores_full.min():.3f}, max={iso_scores_full.max():.3f}")
        # TODO: persist all trained models via ModelRegistry
        for model_name, model_obj in self.models.items():
            metrics = {k: float(v) for k, v in self.results.get(model_name, {}).items() if isinstance(v, (int, float))}
            try:
                self.registry.save(model_name, model_obj, self.feature_cols, metrics, scaler=self.scaler)
            except Exception as exc:
                _logger.warning("ModelRegistry: could not save '%s': %s", model_name, exc)
        best_name, best_entry = self.registry.best_model()
        if best_name:
            _logger.info("ModelRegistry: best model is '%s' (AUC=%.4f)", best_name, best_entry["metrics"].get("auc", 0))
        return self.results

    def _robust_z_scores(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        fill_value = float(numeric.median()) if not numeric.dropna().empty else 0.0
        numeric = numeric.fillna(fill_value)
        median = float(numeric.median())
        mad = float(np.median(np.abs(numeric - median)))
        scale = mad * 1.4826
        if scale < 1e-6:
            scale = float(numeric.std()) or 1.0
        return (numeric - median) / scale

    def _driver_text(self, feature_name: str, row: pd.Series) -> str:
        value = row.get(feature_name, 0)
        if feature_name == "zero_measurement_pct":
            return f"zero-reading share reached {fmt_percent(value, decimals=0)}"
        if feature_name == "sudden_change_ratio":
            return f"abrupt change ratio rose to {float(value):.2f}"
        if feature_name == "night_day_ratio":
            return f"night/day usage ratio reached {float(value):.2f}"
        if feature_name == "peer_consumption_ratio":
            return f"consumption fell to {float(value):.2f}x of the peer baseline"
        if feature_name == "peer_zero_pct_deviation":
            return f"peer zero-reading deviation is {float(value):+.2f}"
        if feature_name == "tamper_density":
            return f"tamper density reached {float(value):.2f} events per month"
        if feature_name == "days_since_last_tamper":
            return f"last tamper signal was only {int(float(value))} days ago"
        if feature_name == "transformer_loss_pct":
            return f"transformer loss climbed to {float(value):.1f}%"
        if feature_name == "load_factor":
            return f"load factor compressed to {float(value):.2f}"
        if feature_name == "rolling_weekly_volatility":
            return f"weekly volatility reached {float(value):.2f}"
        if feature_name == "anomaly_burst_ratio":
            return f"anomaly bursts appeared in {fmt_percent(value, decimals=0)} of weekly windows"
        if feature_name == "outage_event_count":
            return f"{int(float(value))} outage events are masking the signal"
        if feature_name == "meter_age_years":
            return f"meter age is {int(float(value))} years"
        return f"{feature_name.replace('_', ' ')} is elevated"

    def _build_explainability_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        feature_specs = [
            ("zero_measurement_pct", "high"),
            ("sudden_change_ratio", "high"),
            ("night_day_ratio", "high"),
            ("peer_consumption_ratio", "low"),
            ("peer_zero_pct_deviation", "high"),
            ("tamper_density", "high"),
            ("days_since_last_tamper", "low"),
            ("transformer_loss_pct", "high"),
            ("load_factor", "low"),
            ("rolling_weekly_volatility", "high"),
            ("anomaly_burst_ratio", "high"),
            ("outage_event_count", "high"),
            ("meter_age_years", "high"),
        ]
        z_scores: dict[str, pd.Series] = {}
        for feature_name, _direction in feature_specs:
            if feature_name in df.columns:
                z_scores[feature_name] = self._robust_z_scores(df[feature_name])

        reason_1: list[str] = []
        reason_2: list[str] = []
        reason_3: list[str] = []
        drivers: list[str] = []
        summaries: list[str] = []

        for row_index, row in df.iterrows():
            candidates: list[tuple[float, str]] = []
            theft_pattern = safe_text(row.get("theft_type"))
            if theft_pattern not in {"-", "none", "unknown"}:
                candidates.append((4.4, f"pattern aligns with {_humanize_pattern(theft_pattern)}"))
            meter_health = safe_text(row.get("meter_health"))
            if meter_health == "tampered":
                candidates.append((3.8, "meter health indicates tamper risk"))
            elif meter_health == "aging" and float(row.get("meter_age_years", 0) or 0) >= 10:
                candidates.append((2.3, "aging meter drift is compounding the anomaly"))
            if safe_text(row.get("region")) == "rural" and float(row.get("outage_event_count", 0) or 0) >= 2:
                candidates.append((2.0, "repeated outage windows can hide irregular consumption"))
            if float(row.get("est_monthly_loss", 0) or 0) >= 2500:
                candidates.append((2.6, f"monthly exposure is high at {fmt_currency(row.get('est_monthly_loss', 0))}"))

            for feature_name, direction in feature_specs:
                if feature_name not in z_scores:
                    continue
                score = float(z_scores[feature_name].iat[row_index])
                influence = score if direction == "high" else -score
                threshold = 0.7 if feature_name in {"outage_event_count", "meter_age_years"} else 0.85
                if influence > threshold:
                    candidates.append((influence, self._driver_text(feature_name, row)))

            if not candidates:
                candidates.append((1.0, f"fraud probability reached {fmt_percent(row.get('theft_probability', 0), decimals=1)}"))

            chosen: list[str] = []
            for _score, text in sorted(candidates, key=lambda item: item[0], reverse=True):
                if text not in chosen:
                    chosen.append(text)
                if len(chosen) == 3:
                    break
            while len(chosen) < 3:
                chosen.append("-")

            summary_parts = [item for item in chosen if item != "-"]
            risk_band = safe_text(row.get("risk_category"))
            if len(summary_parts) == 1:
                joined = summary_parts[0]
            elif len(summary_parts) == 2:
                joined = f"{summary_parts[0]} and {summary_parts[1]}"
            else:
                joined = f"{summary_parts[0]}, {summary_parts[1]}, and {summary_parts[2]}"
            summary = f"{risk_band} risk ({fmt_percent(row.get('theft_probability', 0), decimals=1)}) driven by {joined}."

            reason_1.append(chosen[0])
            reason_2.append(chosen[1])
            reason_3.append(chosen[2])
            drivers.append(" | ".join(summary_parts))
            summaries.append(summary)

        df["risk_reason_1"] = reason_1
        df["risk_reason_2"] = reason_2
        df["risk_reason_3"] = reason_3
        df["risk_drivers"] = drivers
        df["risk_summary"] = summaries
        return df

    def _build_explainability_snapshot(self, df: pd.DataFrame) -> str:
        if df.empty or "risk_reason_1" not in df.columns:
            return "Explainability signals will appear after scoring."
        focus = df[df["theft_probability"] >= 0.5].copy()
        if focus.empty:
            focus = df.head(20).copy()
        counts: Counter[str] = Counter()
        for column in ["risk_reason_1", "risk_reason_2", "risk_reason_3"]:
            if column in focus.columns:
                counts.update(value for value in focus[column].astype(str) if value and value != "-")
        top = [item for item, _count in counts.most_common(3)]
        if not top:
            return "Explainability signals were not strong enough to summarize."
        if len(top) == 1:
            joined = top[0]
        elif len(top) == 2:
            joined = f"{top[0]} and {top[1]}"
        else:
            joined = f"{top[0]}, {top[1]}, and {top[2]}"
        return f"Most common alert drivers in this run were {joined}."

    def score_customers(self, callback=None):
        if self.df_features is None or self.df_features.empty or not self.models:
            raise ValueError("Scoring requires both a dataset and a trained model.")

        if callback:
            callback(97, "Scoring customers")

        best_model_name = self.best_model_name() if self.results else "Isolation Forest"
        best_model = self.models[best_model_name]
        X = self.df_features[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_scaled)[:, 1]
        else:
            probs = -best_model.score_samples(X_scaled)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

        df = self.df_features.copy()
        df["theft_probability"] = np.round(probs, 4)
        df["risk_score"] = np.round(probs * 100, 1)
        df["risk_category"] = pd.cut(probs, bins=[-1, 0.30, 0.45, 0.70, 0.85, 1.01], labels=RISK_LABELS)
        df["predicted_theft"] = (probs >= 0.5).astype(int)
        base_loss = df.get("mean_consumption", df.get("mean_daily_total", pd.Series(np.ones(len(df)))))
        tariff_rate_map = {"T1": 2.28, "T2": 2.85, "T3": 1.92}
        if "tariff_group" in df.columns:
            tariff_rate = df["tariff_group"].map(tariff_rate_map).fillna(2.15)
            reactive_penalty = np.where(df["tariff_group"] == "T3", 1.12, 1.0)
        else:
            tariff_rate = 2.15
            reactive_penalty = 1.0
        df["est_monthly_loss"] = np.where(probs >= 0.5, np.round(base_loss * 30 * tariff_rate * reactive_penalty * probs, 0), 0)
        df["priority_index"] = np.round(df["risk_score"] * 0.65 + df["est_monthly_loss"].clip(upper=5000) * 0.01, 2)
        df = self._build_explainability_columns(df)

        self.df_scored = df.sort_values(["theft_probability", "priority_index"], ascending=False).reset_index(drop=True)
        self.last_run_at = datetime.now()
        self.last_explainability_summary = self._build_explainability_snapshot(self.df_scored)
        self.log(f"Scoring complete. Selected model: {best_model_name}")
        self.log(f"High-risk customers: {(self.df_scored['theft_probability'] > 0.7).sum()}")
        self.log(f"Estimated monthly exposure: {fmt_currency(self.df_scored['est_monthly_loss'].sum())}")
        self.log(f"Explainability snapshot: {self.last_explainability_summary}")
        return self.df_scored

    def build_overview(self):
        if self.df_scored is None or self.df_scored.empty:
            return None

        df = self.df_scored
        best_model = self.best_model_name()
        return {
            "best_model": best_model,
            "customer_count": int(len(df)),
            "high_risk_count": int((df["theft_probability"] > 0.7).sum()),
            "critical_count": int(df["risk_category"].astype(str).isin(CRITICAL_RISK_LABELS).sum()),
            "average_probability": float(df["theft_probability"].mean()),
            "total_loss": float(df["est_monthly_loss"].sum()),
            "top_customer": safe_text(df.iloc[0]["customer_id"]) if not df.empty else "-",
            "data_source": self.last_source,
            "last_run_at": self.last_run_at.strftime("%H:%M:%S") if self.last_run_at else "-",
            "last_run_at_full": self.last_run_at.isoformat(timespec="minutes") if self.last_run_at else None,
            "preset_name": self.last_preset_name,
            "preset_summary": self.last_preset_summary,
            "explainability_summary": self.last_explainability_summary,
        }
