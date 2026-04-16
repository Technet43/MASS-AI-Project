"""Replay a canonical telemetry CSV into the ingest pipeline.

Primary use: feed `project/data/demo/demo_telemetry.csv` into the
Postgres `raw_telemetry` table so dashboard Telemetry / Live Alerts
panels have data without an external sensor.

If psycopg2 is unavailable or DB is unreachable, falls back to dry-run
and prints the first few rows that would have been inserted. This keeps
demos usable on laptops without the Docker stack running.

Usage:
    python -m project.realtime_ingest.simulate_ingest \
        --file project/data/demo/demo_telemetry.csv \
        [--dry-run] [--limit 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from realtime_ingest.data_loader import normalize_columns, validate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay telemetry CSV into ingest pipeline")
    parser.add_argument("--file", type=Path, required=True, help="Canonical telemetry CSV")
    parser.add_argument("--dry-run", action="store_true", help="Do not connect to Postgres")
    parser.add_argument("--limit", type=int, default=None, help="Only replay first N rows")
    return parser.parse_args(argv)


def load_canonical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    missing = validate(df)
    if missing:
        raise SystemExit(
            f"CSV missing required columns: {missing}. Present: {list(df.columns)}"
        )
    return df


def _insert_postgres(df: pd.DataFrame) -> int:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("[INFO] psycopg2 not installed — falling back to dry-run")
        return _dry_run(df)

    try:
        # Reuse data_loader's insert path for consistency
        from realtime_ingest.data_loader import insert_postgres
    except Exception as exc:
        print(f"[INFO] data_loader insert unavailable ({exc}) — dry-run")
        return _dry_run(df)

    try:
        return insert_postgres(df, dry_run=False)
    except SystemExit as exc:
        print(f"[INFO] Postgres unreachable — dry-run instead ({exc})")
        return _dry_run(df)


def _dry_run(df: pd.DataFrame) -> int:
    preview = df.head(5)
    print("[DRY-RUN] Sample rows that would be inserted:")
    print(preview.to_string(index=False))
    print(f"[DRY-RUN] Total rows that would be inserted: {len(df)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    df = load_canonical(args.file)
    if args.limit is not None:
        df = df.head(args.limit)

    print(f"[INFO] Replaying {len(df)} rows from {args.file}")
    if args.dry_run:
        _dry_run(df)
    else:
        _insert_postgres(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
