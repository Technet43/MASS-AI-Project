import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from app_prefs import load_theme_preference, save_theme_preference
from mass_ai_domain import (
    build_executive_brief_text,
    filter_case_dataframe,
    is_case_overdue,
    priority_for_risk,
)
from ops_store import OpsStore
from support_bundle import create_support_bundle


def make_scored_frame():
    return pd.DataFrame(
        [
            {
                "customer_id": "1001",
                "profile": "residential",
                "theft_type": "night_zeroing",
                "risk_category": "Urgent",
                "theft_probability": 0.92,
                "risk_score": 92.0,
                "est_monthly_loss": 3400,
                "priority_index": 96.2,
            },
            {
                "customer_id": "1002",
                "profile": "commercial",
                "theft_type": "constant_reduction",
                "risk_category": "High",
                "theft_probability": 0.61,
                "risk_score": 61.0,
                "est_monthly_loss": 2100,
                "priority_index": 60.1,
            },
            {
                "customer_id": "1003",
                "profile": "industrial",
                "theft_type": "none",
                "risk_category": "Low",
                "theft_probability": 0.22,
                "risk_score": 22.0,
                "est_monthly_loss": 0,
                "priority_index": 14.3,
            },
        ]
    )


class OpsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "mass_ai_ops.sqlite"
        self.store = OpsStore(db_path=self.db_path)
        self.run_meta = {
            "created_at": datetime.now().isoformat(timespec="minutes"),
            "source_name": "unit-test.csv",
            "model_name": "Isolation Forest",
            "customer_count": 3,
            "high_risk_count": 2,
            "total_exposure": 5500,
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sqlite_initialization_creates_database(self):
        self.assertTrue(self.db_path.exists())

    def test_first_sync_creates_cases_above_threshold(self):
        self.store.sync_run(make_scored_frame(), self.run_meta)
        cases = self.store.list_cases({"status": "All statuses"})
        self.assertEqual(len(cases), 2)
        self.assertEqual(set(cases["customer_id"].astype(str)), {"1001", "1002"})

    def test_rerun_updates_snapshot_without_duplication_and_preserves_manual_state(self):
        self.store.sync_run(make_scored_frame(), self.run_meta)
        self.store.update_case("1001", status="Resolved", priority="P3", follow_up_at="", resolution_reason="False Positive")

        rerun = make_scored_frame().copy()
        rerun.loc[0, "theft_probability"] = 0.21
        rerun.loc[0, "risk_category"] = "Low"
        rerun.loc[0, "est_monthly_loss"] = 0
        rerun.loc[1, "theft_probability"] = 0.88
        rerun.loc[1, "risk_category"] = "Critical"
        self.store.sync_run(rerun, {**self.run_meta, "created_at": datetime.now().isoformat(timespec="minutes")})

        cases = self.store.list_cases({"status": "All statuses"})
        case_1001 = self.store.get_case("1001")
        self.assertEqual(len(cases), 2)
        self.assertEqual(case_1001["status"], "Resolved")
        self.assertEqual(case_1001["priority"], "P3")
        self.assertAlmostEqual(float(case_1001["fraud_probability"]), 0.21, places=2)
        self.assertEqual(case_1001["risk_band"], "Low")

    def test_notes_append_in_chronological_order(self):
        self.store.sync_run(make_scored_frame(), self.run_meta)
        self.store.add_case_note("1001", "First note")
        self.store.add_case_note("1001", "Second note")
        notes = self.store.list_case_notes("1001")
        self.assertEqual([note["note_text"] for note in notes], ["First note", "Second note"])

    def test_case_history_tracks_sync_update_and_notes(self):
        self.store.sync_run(make_scored_frame(), self.run_meta)
        self.store.update_case("1001", status="Escalated", priority="P1", follow_up_at="2026-04-15", resolution_reason="")
        self.store.add_case_note("1001", "Escalation note")
        history = self.store.list_case_history("1001")
        event_types = [item["event_type"] for item in history]
        self.assertIn("case_created", event_types)
        self.assertIn("case_update", event_types)
        self.assertIn("note_added", event_types)


class OpsHelperTests(unittest.TestCase):
    def test_priority_mapping_and_overdue_logic(self):
        self.assertEqual(priority_for_risk("Urgent"), "P1")
        self.assertEqual(priority_for_risk("Moderate"), "P3")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertTrue(is_case_overdue(yesterday, "In Review"))
        self.assertFalse(is_case_overdue(yesterday, "Resolved"))

    def test_queue_filtering_and_report_generation(self):
        df = pd.DataFrame(
            [
                {
                    "customer_id": "1001",
                    "case_title": "Customer 1001 - Critical risk",
                    "profile": "residential",
                    "fraud_pattern": "night_zeroing",
                    "risk_band": "Critical",
                    "fraud_probability": 0.87,
                    "priority": "P1",
                    "status": "In Review",
                    "follow_up_at": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "est_monthly_loss": 3200,
                    "is_overdue": True,
                },
                {
                    "customer_id": "1002",
                    "case_title": "Customer 1002 - High risk",
                    "profile": "commercial",
                    "fraud_pattern": "constant_reduction",
                    "risk_band": "High",
                    "fraud_probability": 0.62,
                    "priority": "P2",
                    "status": "Monitoring",
                    "follow_up_at": "",
                    "est_monthly_loss": 1800,
                    "is_overdue": False,
                },
            ]
        )

        filtered = filter_case_dataframe(
            df,
            search="night",
            status="In Review",
            risk_band="Critical",
            priority="P1",
            overdue_only=True,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["customer_id"], "1001")

        overview = {
            "data_source": "Persistent Ops Center",
            "best_model": "Ops Center snapshot",
            "customer_count": 2,
            "high_risk_count": 2,
            "preset_name": "Turkey Urban",
            "preset_summary": "Dense metro utility mix",
            "explainability_summary": "Most common alert drivers in this run were zero-reading share and tamper density.",
        }
        ops_metrics = {
            "open_cases": 2,
            "overdue": 1,
            "open_by_status": {"New": 0, "In Review": 1, "Escalated": 0, "Monitoring": 1, "Resolved": 0},
        }
        report = build_executive_brief_text(
            overview,
            ops_metrics,
            filtered,
            {
                "case_title": "Customer 1001 - Critical risk",
                "status": "In Review",
                "priority": "P1",
                "risk_band": "Critical",
                "recommended_action": "Escalate immediately.",
                "risk_summary": "Critical risk driven by zero-reading share, tamper density, and peer deviation.",
            },
            [{"created_at": datetime.now().isoformat(timespec="minutes"), "note_text": "Analyst confirmed unusual overnight pattern."}],
        )
        self.assertIn("Open Ops cases: 2", report)
        self.assertIn("Synthetic preset: Turkey Urban", report)
        self.assertIn("Escalate immediately.", report)
        self.assertIn("Why flagged", report)
        self.assertIn("Analyst confirmed unusual overnight pattern.", report)


class AppPrefsTests(unittest.TestCase):
    def test_theme_preference_round_trip_and_invalid_json_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "mass_ai_prefs.json"
            save_theme_preference("Black", prefs_path)
            self.assertEqual(load_theme_preference(prefs_path), "Black")

            prefs_path.write_text("{invalid json", encoding="utf-8")
            self.assertIsNone(load_theme_preference(prefs_path))

    def test_support_bundle_export_creates_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "support.zip"
            created = create_support_bundle(
                bundle_path,
                theme_name="White",
                overview={"best_model": "Isolation Forest", "data_source": "Synthetic dataset"},
                ops_metrics={"open_cases": 2},
                selected_case={"customer_id": "1001"},
                selected_notes=[{"note_text": "hello", "created_at": datetime.now().isoformat(timespec="minutes")}],
                case_history=[{"event_type": "case_created", "event_summary": "created", "created_at": datetime.now().isoformat(timespec="minutes")}],
                log_lines=["line 1", "line 2"],
                current_df=make_scored_frame(),
            )
            self.assertTrue(created.exists())


if __name__ == "__main__":
    unittest.main()
