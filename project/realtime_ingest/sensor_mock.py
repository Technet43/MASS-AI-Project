"""
Local MQTT telemetry producer for MASS-AI.

This mock simulates smart meter telemetry and publishes 5 messages per second
to the `mass_ai/telemetry` topic.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:  # pragma: no cover - import guard for runtime
    raise SystemExit(
        "Missing dependency: paho-mqtt\n"
        "Install it with: python -m pip install paho-mqtt"
    ) from exc


BROKER_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mass_ai/telemetry")
PUBLISH_RATE_HZ = 5
QOS = 1


def utc_now_iso() -> str:
    """Return the current UTC timestamp in RFC3339 / ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class MeterProfile:
    meter_id: str
    profile: str
    base_voltage: float
    base_power_watts: float
    power_factor: float
    daily_phase: float = field(default_factory=lambda: random.uniform(0.0, math.tau))
    noise_scale: float = 0.04

    def generate_reading(self, now_epoch: float) -> dict[str, float | str]:
        """
        Create one realistic telemetry frame.

        The power curve uses a daily sinus pattern plus a smaller short-wave
        signal and random noise so the values look alive without being chaotic.
        """
        hour_of_day = (now_epoch % 86400) / 3600.0
        daily_wave = 0.55 + 0.45 * math.sin((hour_of_day / 24.0) * math.tau + self.daily_phase)
        burst_wave = 0.06 * math.sin(now_epoch / 45.0 + self.daily_phase)
        stochastic_noise = random.uniform(-self.noise_scale, self.noise_scale)

        power_multiplier = max(0.12, daily_wave + burst_wave + stochastic_noise)
        active_power = self.base_power_watts * power_multiplier

        voltage_drift = random.uniform(-5.0, 5.0)
        voltage = max(205.0, min(245.0, self.base_voltage + voltage_drift))
        current = active_power / max(voltage * self.power_factor, 1.0)

        return {
            "meter_id": self.meter_id,
            "timestamp": utc_now_iso(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "active_power": round(active_power, 2),
        }


def build_meter_fleet(size: int = 60) -> list[MeterProfile]:
    """Build a mixed fleet of residential, commercial, and industrial meters."""
    fleet: list[MeterProfile] = []
    for index in range(size):
        if index < int(size * 0.65):
            profile = "residential"
            base_power = random.uniform(450.0, 4200.0)
            voltage = 230.0
            power_factor = random.uniform(0.93, 0.99)
        elif index < int(size * 0.90):
            profile = "commercial"
            base_power = random.uniform(3500.0, 24000.0)
            voltage = 230.0
            power_factor = random.uniform(0.90, 0.97)
        else:
            profile = "industrial"
            base_power = random.uniform(18000.0, 85000.0)
            voltage = 400.0
            power_factor = random.uniform(0.88, 0.96)

        fleet.append(
            MeterProfile(
                meter_id=f"METER-{index + 1:05d}",
                profile=profile,
                base_voltage=voltage,
                base_power_watts=base_power,
                power_factor=power_factor,
            )
        )
    return fleet


class SensorMockPublisher:
    """High-level wrapper for MQTT connection management and telemetry publishing."""

    def __init__(self) -> None:
        self.running = True
        self.client = mqtt.Client(
            client_id=f"mass-ai-sensor-mock-{os.getpid()}",
            protocol=mqtt.MQTTv311,
        )
        self.client.enable_logger()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.loop_start()
        self.meters = build_meter_fleet()

    def stop(self, *_args: object) -> None:
        """Request a graceful shutdown."""
        self.running = False

    @staticmethod
    def on_connect(_client: mqtt.Client, _userdata: object, _flags: dict[str, int], rc: int) -> None:
        if rc == 0:
            print(f"[sensor_mock] connected to {BROKER_HOST}:{BROKER_PORT}", flush=True)
            return
        print(f"[sensor_mock] connect failed with code={rc}", file=sys.stderr, flush=True)

    @staticmethod
    def on_disconnect(_client: mqtt.Client, _userdata: object, rc: int) -> None:
        if rc == 0:
            print("[sensor_mock] disconnected cleanly", flush=True)
            return
        print(f"[sensor_mock] disconnected unexpectedly code={rc}", file=sys.stderr, flush=True)

    def connect(self) -> None:
        """Connect to MQTT broker with clear error handling."""
        try:
            self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
        except OSError as exc:
            raise SystemExit(
                f"Unable to connect to MQTT broker at {BROKER_HOST}:{BROKER_PORT}: {exc}"
            ) from exc

    def publish_loop(self) -> None:
        """Publish one message every 200ms to keep throughput at 5 msg/sec."""
        interval = 1.0 / PUBLISH_RATE_HZ
        next_tick = time.monotonic()

        while self.running:
            meter = random.choice(self.meters)
            payload = meter.generate_reading(time.time())

            try:
                message = json.dumps(payload, separators=(",", ":"))
                result = self.client.publish(MQTT_TOPIC, payload=message, qos=QOS)
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    print(
                        f"[sensor_mock] publish failed rc={result.rc} meter={payload['meter_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[sensor_mock] published {message}", flush=True)
            except (TypeError, ValueError) as exc:
                print(f"[sensor_mock] serialization error: {exc}", file=sys.stderr, flush=True)
            except Exception as exc:  # pragma: no cover - runtime safety net
                print(f"[sensor_mock] unexpected publish error: {exc}", file=sys.stderr, flush=True)

            next_tick += interval
            sleep_for = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_for)

    def close(self) -> None:
        """Stop MQTT network loop cleanly."""
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass


def main() -> int:
    publisher = SensorMockPublisher()
    signal.signal(signal.SIGINT, publisher.stop)
    signal.signal(signal.SIGTERM, publisher.stop)
    publisher.connect()

    try:
        publisher.publish_loop()
    finally:
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
