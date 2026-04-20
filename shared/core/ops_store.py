from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

CASE_THRESHOLD = 0.50
CASE_STATUSES = ["New", "In Review", "Escalated", "Monitoring", "Resolved"]
CASE_PRIORITIES = ["P1", "P2", "P3", "P4"]
RESOLUTION_REASONS = [
    "False Positive",
    "Field Check Completed",
    "Customer Contacted",
    "Monitoring Only",
    "Other",
]
DEFAULT_PRIORITY_BY_RISK = {
    "Urgent": "P1",
    "Critical": "P1",
    "High": "P2",
    "Moderate": "P3",
    "Low": "P4",
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_ops_db_path() -> Path:
    if os.name == "nt":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "MASS-AI"
    else:
        base_dir = Path.home() / ".mass-ai"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "mass_ai_ops.sqlite"


def priority_for_risk_band(risk_band: Any) -> str:
    return DEFAULT_PRIORITY_BY_RISK.get(str(risk_band or "").strip(), "P4")


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", "-"):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def is_overdue_value(follow_up_at: Any, status: Any, reference: datetime | None = None) -> bool:
    if str(status or "").strip() == "Resolved":
        return False
    follow_up_dt = parse_datetime(follow_up_at)
    if follow_up_dt is None:
        return False
    now_dt = reference or datetime.now()
    return follow_up_dt.date() < now_dt.date()


def case_columns() -> list[str]:
    return [
        "customer_id",
        "case_title",
        "profile",
        "fraud_pattern",
        "risk_band",
        "fraud_probability",
        "risk_score",
        "est_monthly_loss",
        "priority_index",
        "recommended_action",
        "status",
        "priority",
        "follow_up_at",
        "resolution_reason",
        "created_at",
        "updated_at",
        "last_analysis_at",
        "last_seen_run_id",
        "source_name",
        "is_overdue",
    ]


class OpsStore:
    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = Path(db_path) if db_path else resolve_ops_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _managed_connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def init_db(self) -> None:
        with self._managed_connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_name TEXT,
                    model_name TEXT,
                    customer_count INTEGER,
                    high_risk_count INTEGER,
                    total_exposure REAL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    customer_id TEXT PRIMARY KEY,
                    case_title TEXT NOT NULL,
                    profile TEXT,
                    fraud_pattern TEXT,
                    risk_band TEXT,
                    fraud_probability REAL,
                    risk_score REAL,
                    est_monthly_loss REAL,
                    priority_index REAL,
                    recommended_action TEXT,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    follow_up_at TEXT,
                    resolution_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_analysis_at TEXT,
                    last_seen_run_id INTEGER,
                    source_name TEXT,
                    FOREIGN KEY(last_seen_run_id) REFERENCES analysis_runs(id)
                );

                CREATE TABLE IF NOT EXISTS case_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES cases(customer_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS case_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES cases(customer_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
                CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
                CREATE INDEX IF NOT EXISTS idx_cases_risk_band ON cases(risk_band);
                CREATE INDEX IF NOT EXISTS idx_case_notes_customer ON case_notes(customer_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_case_history_customer ON case_history(customer_id, created_at);

                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    inspection_date TEXT NOT NULL,
                    result TEXT NOT NULL,
                    inspector_notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES cases(customer_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_inspections_customer ON inspections(customer_id, inspection_date);
                """
            )
            # Default users seed
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM users")
            if cur.fetchone()["cnt"] == 0:
                # Seed default users
                cur.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'admin', 'Admin')")
                cur.execute("INSERT INTO users (username, password, role) VALUES ('analyst', 'analyst', 'Analyst')")
                cur.execute("INSERT INTO users (username, password, role) VALUES ('field', 'field', 'Field_Ops')")
                conn.commit()

    def authenticate(self, username: str, password: str) -> dict[str, str] | None:
        with self._managed_connection() as conn:
            user = conn.execute(
                "SELECT username, role FROM users WHERE username = ? AND password = ?", 
                (username, password)
            ).fetchone()
            if user:
                return {"username": user["username"], "role": user["role"]}
            return None

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        data["is_overdue"] = is_overdue_value(data.get("follow_up_at"), data.get("status"))
        return data

    def _recommended_action(self, risk_band: Any) -> str:
        risk_text = str(risk_band or "").strip()
        if risk_text in {"Urgent", "Critical"}:
            return "Escalate for field inspection and dispatch this case into the active verification queue."
        if risk_text == "High":
            return "Send the case to operations review and schedule a secondary validation check."
        return "Keep the account in the monitoring queue and rescore it when new data arrives."

    def _append_case_history(self, cursor: sqlite3.Cursor, customer_id: str, event_type: str, event_summary: str, created_at: str | None = None) -> None:
        cursor.execute(
            "INSERT INTO case_history (customer_id, event_type, event_summary, created_at) VALUES (?, ?, ?, ?)",
            (customer_id, event_type, event_summary, created_at or now_text()),
        )

    def sync_run(self, scored_df: pd.DataFrame, run_meta: dict[str, Any]) -> int:
        if scored_df is None or scored_df.empty:
            raise ValueError("A scored dataframe is required to sync the Ops Center queue.")

        created_at = str(run_meta.get("created_at") or now_text())
        source_name = str(run_meta.get("source_name") or "Unknown source")
        model_name = str(run_meta.get("model_name") or "Unknown model")
        customer_count = int(run_meta.get("customer_count") or len(scored_df))
        high_risk_count = int(run_meta.get("high_risk_count") or 0)
        total_exposure = float(run_meta.get("total_exposure") or 0.0)

        with self._managed_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO analysis_runs (
                    created_at, source_name, model_name, customer_count, high_risk_count, total_exposure
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (created_at, source_name, model_name, customer_count, high_risk_count, total_exposure),
            )
            run_id = int(cur.lastrowid)

            for _, row in scored_df.iterrows():
                customer_id = str(row.get("customer_id", "")).strip()
                if not customer_id:
                    continue

                existing = cur.execute(
                    "SELECT customer_id, status, priority, follow_up_at, resolution_reason, created_at FROM cases WHERE customer_id = ?",
                    (customer_id,),
                ).fetchone()
                probability = float(row.get("theft_probability", 0) or 0)
                if existing is None and probability < CASE_THRESHOLD:
                    continue

                risk_band = str(row.get("risk_category", "Low") or "Low")
                current_status = existing["status"] if existing else "New"
                current_priority = existing["priority"] if existing and existing["priority"] else priority_for_risk_band(risk_band)
                current_follow_up = existing["follow_up_at"] if existing else None
                current_resolution = existing["resolution_reason"] if existing else None
                created_value = existing["created_at"] if existing else created_at

                payload = (
                    f"Customer {customer_id} - {risk_band} risk",
                    str(row.get("profile", "unknown") or "unknown"),
                    str(row.get("theft_type", "unknown") or "unknown"),
                    risk_band,
                    probability,
                    float(row.get("risk_score", 0) or 0),
                    float(row.get("est_monthly_loss", 0) or 0),
                    float(row.get("priority_index", 0) or 0),
                    self._recommended_action(risk_band),
                    current_status,
                    current_priority,
                    current_follow_up,
                    current_resolution,
                    created_value,
                    now_text(),
                    created_at,
                    run_id,
                    source_name,
                    customer_id,
                )

                if existing:
                    cur.execute(
                        """
                        UPDATE cases
                        SET case_title = ?,
                            profile = ?,
                            fraud_pattern = ?,
                            risk_band = ?,
                            fraud_probability = ?,
                            risk_score = ?,
                            est_monthly_loss = ?,
                            priority_index = ?,
                            recommended_action = ?,
                            status = ?,
                            priority = ?,
                            follow_up_at = ?,
                            resolution_reason = ?,
                            created_at = ?,
                            updated_at = ?,
                            last_analysis_at = ?,
                            last_seen_run_id = ?,
                            source_name = ?
                        WHERE customer_id = ?
                        """,
                        payload,
                    )
                    self._append_case_history(
                        cur,
                        customer_id,
                        "analysis_sync",
                        f"Snapshot refreshed from {source_name}: {risk_band} risk at {probability:.2f} probability.",
                        created_at,
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO cases (
                            case_title, profile, fraud_pattern, risk_band, fraud_probability, risk_score,
                            est_monthly_loss, priority_index, recommended_action, status, priority,
                            follow_up_at, resolution_reason, created_at, updated_at, last_analysis_at,
                            last_seen_run_id, source_name, customer_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        payload,
                    )
                    self._append_case_history(
                        cur,
                        customer_id,
                        "case_created",
                        f"Case created from {source_name}: {risk_band} risk at {probability:.2f} probability.",
                        created_at,
                    )

            conn.commit()
        return run_id

    def list_cases(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        filters = filters or {}
        with self._managed_connection() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM cases")]

        if not rows:
            return pd.DataFrame(columns=case_columns())

        df = pd.DataFrame(rows)
        df["is_overdue"] = df.apply(lambda row: is_overdue_value(row.get("follow_up_at"), row.get("status")), axis=1)

        search = str(filters.get("search") or "").strip().lower()
        if search:
            mask = pd.Series(False, index=df.index)
            for column in ["customer_id", "profile", "fraud_pattern", "case_title"]:
                if column in df.columns:
                    mask = mask | df[column].astype(str).str.lower().str.contains(search, na=False)
            df = df[mask]

        status = str(filters.get("status") or "").strip()
        if status and status not in {"All statuses", "All open"}:
            df = df[df["status"].astype(str) == status]
        elif status == "All open":
            df = df[df["status"].astype(str) != "Resolved"]

        risk_band = str(filters.get("risk_band") or "").strip()
        if risk_band and risk_band != "All risk bands":
            df = df[df["risk_band"].astype(str) == risk_band]

        priority = str(filters.get("priority") or "").strip()
        if priority and priority != "All priorities":
            df = df[df["priority"].astype(str) == priority]

        if bool(filters.get("overdue_only")):
            df = df[df["is_overdue"]]

        priority_order = {value: idx for idx, value in enumerate(CASE_PRIORITIES)}
        status_order = {value: idx for idx, value in enumerate(CASE_STATUSES)}
        df["priority_rank"] = df["priority"].map(priority_order).fillna(len(priority_order))
        df["status_rank"] = df["status"].map(status_order).fillna(len(status_order))
        df["follow_up_sort"] = df["follow_up_at"].fillna("9999-12-31T23:59:59")
        df = df.sort_values(
            ["is_overdue", "priority_rank", "fraud_probability", "priority_index", "follow_up_sort", "updated_at"],
            ascending=[False, True, False, False, True, False],
            kind="mergesort",
        ).drop(columns=["priority_rank", "status_rank", "follow_up_sort"])
        return df.reset_index(drop=True)

    def get_case(self, customer_id: Any) -> dict[str, Any] | None:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            return None

        with self._managed_connection() as conn:
            row = conn.execute("SELECT * FROM cases WHERE customer_id = ?", (customer_key,)).fetchone()
        return self._row_to_dict(row)

    def update_case(
        self,
        customer_id: Any,
        status: str | None = None,
        priority: str | None = None,
        follow_up_at: str | None = None,
        resolution_reason: str | None = None,
    ) -> None:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            raise ValueError("A valid customer_id is required to update a case.")

        current_case = self.get_case(customer_key)
        if current_case is None:
            raise ValueError("Case not found.")

        updates: list[str] = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        if follow_up_at is not None:
            updates.append("follow_up_at = ?")
            params.append(follow_up_at or None)
        if resolution_reason is not None:
            updates.append("resolution_reason = ?")
            params.append(resolution_reason or None)

        updates.append("updated_at = ?")
        params.append(now_text())
        params.append(customer_key)

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE cases SET {', '.join(updates)} WHERE customer_id = ?", params)
            changes: list[str] = []
            if status is not None and status != current_case.get("status"):
                changes.append(f"status {current_case.get('status')} -> {status}")
            if priority is not None and priority != current_case.get("priority"):
                changes.append(f"priority {current_case.get('priority')} -> {priority}")
            if follow_up_at is not None and (follow_up_at or None) != current_case.get("follow_up_at"):
                changes.append(f"follow-up -> {follow_up_at or 'cleared'}")
            if resolution_reason is not None and (resolution_reason or None) != current_case.get("resolution_reason"):
                changes.append(f"resolution -> {resolution_reason or 'cleared'}")
            if changes:
                self._append_case_history(cursor, customer_key, "case_update", "; ".join(changes))
            conn.commit()

    def add_case_note(self, customer_id: Any, note_text: str) -> None:
        customer_key = str(customer_id or "").strip()
        note = str(note_text or "").strip()
        if not customer_key:
            raise ValueError("A valid customer_id is required to add a note.")
        if not note:
            raise ValueError("A note cannot be empty.")

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO case_notes (customer_id, note_text, created_at) VALUES (?, ?, ?)",
                (customer_key, note, now_text()),
            )
            self._append_case_history(cursor, customer_key, "note_added", f"Analyst note added: {note[:96]}")
            conn.commit()

    def list_case_notes(self, customer_id: Any) -> list[dict[str, Any]]:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            return []

        with self._managed_connection() as conn:
            rows = conn.execute(
                "SELECT id, customer_id, note_text, created_at FROM case_notes WHERE customer_id = ? ORDER BY created_at ASC, id ASC",
                (customer_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def case_metrics(self) -> dict[str, Any]:
        cases = self.list_cases()
        if cases.empty:
            return {
                "open_cases": 0,
                "urgent": 0,
                "overdue": 0,
                "resolved_this_week": 0,
                "open_by_status": {status: 0 for status in CASE_STATUSES},
                "top_priority_open_cases": [],
            }

        open_cases_df = cases[cases["status"].astype(str) != "Resolved"].copy()
        resolved_cutoff = datetime.now() - timedelta(days=7)
        resolved_this_week = cases[
            (cases["status"].astype(str) == "Resolved")
            & (cases["updated_at"].astype(str).apply(lambda value: parse_datetime(value) or datetime.min) >= resolved_cutoff)
        ]

        open_by_status = {status: 0 for status in CASE_STATUSES}
        if not open_cases_df.empty:
            for status, count in open_cases_df["status"].astype(str).value_counts().items():
                open_by_status[status] = int(count)

        priority_rank = {value: idx for idx, value in enumerate(CASE_PRIORITIES)}
        top_priority_open = []
        if not open_cases_df.empty:
            ranked = open_cases_df.copy()
            ranked["priority_rank"] = ranked["priority"].map(priority_rank).fillna(len(priority_rank))
            ranked = ranked.sort_values(
                ["priority_rank", "is_overdue", "fraud_probability", "est_monthly_loss"],
                ascending=[True, False, False, False],
                kind="mergesort",
            ).drop(columns=["priority_rank"])
            top_priority_open = ranked.head(5).to_dict(orient="records")

        return {
            "open_cases": int(len(open_cases_df)),
            "urgent": int((open_cases_df["risk_band"].astype(str) == "Urgent").sum()),
            "overdue": int(open_cases_df["is_overdue"].sum()),
            "resolved_this_week": int(len(resolved_this_week)),
            "open_by_status": open_by_status,
            "top_priority_open_cases": top_priority_open,
        }

    def list_case_history(self, customer_id: Any, limit: int | None = None) -> list[dict[str, Any]]:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            return []
        query = "SELECT id, customer_id, event_type, event_summary, created_at FROM case_history WHERE customer_id = ? ORDER BY created_at ASC, id ASC"
        params: list[Any] = [customer_key]
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def add_inspection(
        self,
        customer_id: Any,
        result: str,
        inspector_notes: str = "",
        inspection_date: str | None = None,
    ) -> None:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            raise ValueError("A valid customer_id is required.")
        if result not in {"clean", "confirmed_theft", "inconclusive"}:
            raise ValueError("Result must be 'clean', 'confirmed_theft', or 'inconclusive'.")
        date_value = inspection_date or datetime.now().strftime("%Y-%m-%d")
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO inspections (customer_id, inspection_date, result, inspector_notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (customer_key, date_value, result, inspector_notes.strip(), now_text()),
            )
            self._append_case_history(cursor, customer_key, "inspection", f"Inspection logged: {result} on {date_value}.")
            conn.commit()

    def list_inspections(self, customer_id: Any) -> list[dict[str, Any]]:
        customer_key = str(customer_id or "").strip()
        if not customer_key:
            return []
        with self._managed_connection() as conn:
            rows = conn.execute(
                "SELECT id, customer_id, inspection_date, result, inspector_notes, created_at FROM inspections WHERE customer_id = ? ORDER BY inspection_date DESC, id DESC",
                (customer_key,),
            ).fetchall()
        return [dict(row) for row in rows]
