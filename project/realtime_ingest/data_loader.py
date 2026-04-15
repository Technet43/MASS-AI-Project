"""
MASS-AI Data Loader
===================
Mühendisler için kolay veri yükleme aracı.

Desteklenen formatlar: CSV, Excel (.xlsx)

Kullanım:
    python data_loader.py                        # Sihirbaz modu
    python data_loader.py --file verim.csv       # Direkt dosya
    python data_loader.py --mqtt                 # MQTT üzerinden gönder
    python data_loader.py --template             # Örnek CSV oluştur

Gerekli sütunlar (en az):
    meter_id, voltage, current, active_power

İsteğe bağlı:
    timestamp  (yoksa şimdiki zaman kullanılır)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Bağımlılık kontrolleri ────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("❌  psycopg2 kurulu değil → pip install psycopg2-binary")

try:
    import pandas as pd
except ImportError:
    sys.exit("❌  pandas kurulu değil → pip install pandas openpyxl")

# ── Ayarlar ───────────────────────────────────────────────────────────────────
DB_CFG = dict(
    host    =os.getenv("DB_HOST",     "localhost"),
    port    =int(os.getenv("DB_PORT", "5433")),
    dbname  =os.getenv("DB_NAME",     "mass_ai"),
    user    =os.getenv("DB_USER",     "mass_ai"),
    password=os.getenv("DB_PASSWORD", "mass_ai_secret"),
)

MQTT_HOST  = os.getenv("MQTT_HOST",  "localhost")
MQTT_PORT  = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mass_ai/telemetry")

REQUIRED_COLS = {"meter_id", "voltage", "current", "active_power"}

# Sütun adı eşleştirme (farklı isimler kullanan dosyalar için)
COL_ALIASES = {
    "sayac_id": "meter_id",  "meter": "meter_id",   "id": "meter_id",
    "gerilim":  "voltage",   "v":     "voltage",
    "akim":     "current",   "i":     "current",     "a": "current",
    "guc":      "active_power", "power": "active_power", "p": "active_power",
    "zaman":    "timestamp", "time":  "timestamp",   "tarih": "timestamp",
}

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def banner():
    print("""
╔══════════════════════════════════════════════╗
║        MASS-AI  Veri Yükleme Aracı           ║
║  CSV / Excel → Postgres raw_telemetry        ║
╚══════════════════════════════════════════════╝
""")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını küçük harfe çevir ve bilinen takma adları eşleştir."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns=COL_ALIASES, inplace=True)
    return df

def validate(df: pd.DataFrame) -> list[str]:
    """Eksik sütunları listele."""
    missing = REQUIRED_COLS - set(df.columns)
    return list(missing)

def preview(df: pd.DataFrame, n: int = 5):
    print(f"\n📋 Önizleme ({min(n, len(df))}/{len(df)} satır):")
    print(df.head(n).to_string(index=False))
    print()

def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        sys.exit(f"❌  Desteklenmeyen format: {suffix}  (CSV veya Excel kullan)")
    return normalize_columns(df)

def connect_db():
    try:
        conn = psycopg2.connect(**DB_CFG)
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        sys.exit(f"❌  Postgres bağlantısı başarısız:\n    {e}\n\n"
                 "    Docker çalışıyor mu?  →  docker compose up -d mass-ai-db")

def insert_postgres(df: pd.DataFrame, dry_run: bool = False) -> int:
    """DataFrame'i raw_telemetry tablosuna toplu INSERT et."""
    conn = connect_db()
    rows_inserted = 0
    sql = """
        INSERT INTO raw_telemetry (meter_id, voltage, current, active_power, received_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    now = datetime.now(timezone.utc)

    records = []
    for _, row in df.iterrows():
        ts = row.get("timestamp", None)
        if pd.isna(ts) or ts is None or str(ts).strip() == "":
            ts = now
        elif not isinstance(ts, datetime):
            try:
                ts = pd.to_datetime(ts, utc=True).to_pydatetime()
            except Exception:
                ts = now
        records.append((
            str(row["meter_id"]),
            float(row["voltage"]),
            float(row["current"]),
            float(row["active_power"]),
            ts,
        ))

    if dry_run:
        print(f"🔍  Kuru çalıştırma: {len(records)} satır yazılacaktı (DB'ye dokunulmadı).")
        conn.close()
        return 0

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
        conn.commit()
        rows_inserted = len(records)
        print(f"✅  {rows_inserted} satır raw_telemetry tablosuna yazıldı.")
    except Exception as e:
        conn.rollback()
        print(f"❌  INSERT hatası: {e}")
    finally:
        conn.close()
    return rows_inserted

def send_mqtt(df: pd.DataFrame, rate_hz: float = 10.0):
    """DataFrame satırlarını MQTT'e gönder."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("❌  paho-mqtt kurulu değil → pip install paho-mqtt")

    client = mqtt.Client(client_id=f"mass-ai-loader-{os.getpid()}")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    interval = 1.0 / rate_hz
    now_iso  = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    sent = 0

    print(f"📡  MQTT gönderimi başlıyor  ({rate_hz} msg/sn)  Ctrl+C ile durdur\n")
    try:
        for _, row in df.iterrows():
            ts = row.get("timestamp", now_iso)
            if pd.isna(ts):
                ts = now_iso
            payload = json.dumps({
                "meter_id":    str(row["meter_id"]),
                "timestamp":   str(ts),
                "voltage":     float(row["voltage"]),
                "current":     float(row["current"]),
                "active_power":float(row["active_power"]),
            }, separators=(",", ":"))
            result = client.publish(MQTT_TOPIC, payload, qos=1)
            result.wait_for_publish()
            sent += 1
            print(f"  [{sent:>5}] {payload[:80]}")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
    print(f"\n✅  {sent} mesaj gönderildi.")

def create_template():
    """Örnek CSV şablonu oluştur."""
    path = Path("ornek_veri.csv")
    rows = [
        ["meter_id",       "timestamp",              "voltage", "current", "active_power"],
        ["METER-00001", "2026-04-15T10:00:00Z",      230.1,     4.2,       970.0],
        ["METER-00001", "2026-04-15T10:01:00Z",      229.8,     4.3,       985.0],
        ["METER-00002", "2026-04-15T10:00:00Z",      231.5,     12.1,      2800.0],
        ["METER-00002", "2026-04-15T10:01:00Z",      232.0,     11.9,      2756.0],
        ["METER-00003", "2026-04-15T10:00:00Z",      228.0,     0.1,       23.0],   # şüpheli
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"✅  Şablon oluşturuldu: {path.resolve()}")
    print("    Doldur ve şunu çalıştır:")
    print(f"    python data_loader.py --file {path.name}")

# ── Sihirbaz modu ─────────────────────────────────────────────────────────────

def wizard():
    banner()
    print("Dosya yolunu gir (CSV veya Excel):")
    path_str = input("  > ").strip().strip('"').strip("'")
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"❌  Dosya bulunamadı: {path}")

    df = load_file(path)
    missing = validate(df)
    if missing:
        print(f"\n❌  Eksik sütunlar: {missing}")
        print(f"    Mevcut sütunlar: {list(df.columns)}")
        sys.exit(1)

    print(f"\n✅  {len(df)} satır okundu.")
    preview(df)

    print("Nereye göndermek istiyorsun?")
    print("  1 → Postgres (doğrudan DB)")
    print("  2 → MQTT (gateway üzerinden canlı akış)")
    print("  3 → İkisi de")
    choice = input("  Seçim (1/2/3): ").strip()

    dry = input("Kuru çalıştırma? DB'ye yazmadan test et (e/h): ").strip().lower() == "e"

    if choice in ("1", "3"):
        insert_postgres(df, dry_run=dry)
    if choice in ("2", "3"):
        rate = input("Gönderim hızı (msg/sn, varsayılan 5): ").strip()
        rate = float(rate) if rate else 5.0
        if not dry:
            send_mqtt(df, rate_hz=rate)
        else:
            print("🔍  Kuru çalıştırma: MQTT gönderimi atlandı.")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MASS-AI Veri Yükleme Aracı")
    parser.add_argument("--file",     help="CSV veya Excel dosyası")
    parser.add_argument("--mqtt",     action="store_true", help="MQTT üzerinden gönder")
    parser.add_argument("--rate",     type=float, default=5.0, help="MQTT gönderim hızı (msg/sn)")
    parser.add_argument("--dry-run",  action="store_true", help="DB'ye yazmadan test et")
    parser.add_argument("--template", action="store_true", help="Örnek CSV şablonu oluştur")
    args = parser.parse_args()

    if args.template:
        create_template()
        return

    if not args.file:
        wizard()
        return

    banner()
    path = Path(args.file)
    if not path.exists():
        sys.exit(f"❌  Dosya bulunamadı: {path}")

    df = load_file(path)
    missing = validate(df)
    if missing:
        print(f"❌  Eksik sütunlar: {missing}")
        print(f"   Mevcut sütunlar: {list(df.columns)}")
        sys.exit(1)

    print(f"✅  {len(df)} satır okundu — {path.name}")
    preview(df)

    if args.mqtt:
        if not args.dry_run:
            send_mqtt(df, rate_hz=args.rate)
        else:
            print("🔍  Kuru çalıştırma: MQTT gönderimi atlandı.")
    else:
        insert_postgres(df, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
