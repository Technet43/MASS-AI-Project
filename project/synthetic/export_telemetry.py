"""Export synthetic data in canonical formats.

Single generator run → two outputs:
  - long-format telemetry CSV (canonical ingest contract)
  - feature-table CSV (batch ML pipeline)

Usage:
    python -m project.synthetic.export_telemetry \
        --customers 100 --days 7 \
        --telemetry-output project/data/demo/demo_telemetry.csv \
        --feature-output   project/data/demo/demo_feature_table.csv \
        --manifest         project/data/demo/demo_manifest.json \
        --preset "Turkey Urban" --seed 20260416

At least one of --telemetry-output / --feature-output must be provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mass_ai_engine import MassAIEngine, SYNTHETIC_PRESETS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASS-AI synthetic data exporter")
    parser.add_argument("--customers", type=int, default=100, help="Number of customers (meters)")
    parser.add_argument("--days", type=int, default=7, help="Days of hourly history per meter")
    parser.add_argument(
        "--preset",
        type=str,
        default="Turkey Urban",
        choices=list(SYNTHETIC_PRESETS.keys()),
        help="Synthetic preset profile",
    )
    parser.add_argument(
        "--start-ts",
        type=str,
        default=None,
        help="ISO-8601 start timestamp (UTC). Defaults to now - days.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    parser.add_argument(
        "--telemetry-output",
        type=Path,
        default=None,
        help="Destination CSV for long-format telemetry",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=None,
        help="Destination CSV for feature-table rows",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Destination JSON summarising the run (seed, preset, known theft cases)",
    )
    return parser.parse_args(argv)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.telemetry_output is None and args.feature_output is None:
        raise SystemExit("Provide at least one of --telemetry-output / --feature-output")

    want_telemetry = args.telemetry_output is not None
    engine = MassAIEngine()
    features = engine.generate_synthetic(
        n_customers=args.customers,
        n_days=args.days,
        preset_name=args.preset,
        emit_telemetry=want_telemetry,
        telemetry_start_ts=args.start_ts,
        seed=args.seed,
    )

    written: dict[str, str] = {}

    if args.feature_output is not None:
        _ensure_parent(args.feature_output)
        features.to_csv(args.feature_output, index=False)
        written["feature_table"] = str(args.feature_output)

    if want_telemetry:
        telemetry = engine.to_long_telemetry()
        _ensure_parent(args.telemetry_output)
        telemetry_out = telemetry.copy()
        telemetry_out["timestamp"] = telemetry_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        telemetry_out.to_csv(args.telemetry_output, index=False)
        written["telemetry"] = str(args.telemetry_output)

    known_theft = []
    if "label" in features.columns:
        theft_rows = features[features["label"] == 1]
        for _, row in theft_rows.iterrows():
            meter_id = f"METER-{int(row['customer_id']) + 1:05d}"
            known_theft.append(
                {
                    "meter_id": meter_id,
                    "customer_id": int(row["customer_id"]),
                    "theft_type": str(row.get("theft_type", "unknown")),
                    "profile": str(row.get("profile", "")),
                    "region": str(row.get("region", "")),
                }
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": args.preset,
        "customers": int(args.customers),
        "days": int(args.days),
        "seed": args.seed,
        "start_ts": args.start_ts,
        "outputs": written,
        "telemetry_summary": engine.last_telemetry_summary,
        "known_theft_cases": known_theft,
    }

    if args.manifest is not None:
        _ensure_parent(args.manifest)
        with args.manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        written["manifest"] = str(args.manifest)

    print("[OK] Synthetic export complete")
    for kind, path in written.items():
        print(f"   - {kind}: {path}")
    if engine.last_telemetry_summary:
        s = engine.last_telemetry_summary
        print(f"   - telemetry rows: {s['rows']} across {s['meters']} meters ({s['start_ts']} -> {s['end_ts']})")
    print(f"   - known theft cases: {len(known_theft)}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
