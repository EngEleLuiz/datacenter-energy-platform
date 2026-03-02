"""
ingestion/kafka_producer.py

Orchestrates all simulators and streams their telemetry to Kafka topics.
Each topic maps to a Bronze-layer landing zone in MinIO / S3.

Topics:
  dc.telemetry.servers    → ServerTelemetry (100 msgs per tick)
  dc.telemetry.ups        → UPSTelemetry
  dc.telemetry.inverters  → InverterTelemetry
  dc.telemetry.weather    → WeatherReading (every 15 min via API)

Run:
  python -m ingestion.kafka_producer
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone

from confluent_kafka import Producer, KafkaException
from dotenv import load_dotenv
from loguru import logger

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_generator.server_simulator    import ServerSimulator
from data_generator.ups_inverter_simulator import UPSSimulator, InverterSimulator

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC_SERVERS     = os.getenv("KAFKA_TOPIC_SERVERS",    "dc.telemetry.servers")
TOPIC_UPS         = os.getenv("KAFKA_TOPIC_UPS",        "dc.telemetry.ups")
TOPIC_INVERTERS   = os.getenv("KAFKA_TOPIC_INVERTERS",  "dc.telemetry.inverters")
TOPIC_WEATHER     = os.getenv("KAFKA_TOPIC_WEATHER",    "dc.telemetry.weather")
INTERVAL_SECONDS  = float(os.getenv("DC_SIMULATION_INTERVAL_SECONDS", "5"))
WEATHER_INTERVAL  = 900  # 15 minutes — respects Open-Meteo free tier

NUM_SERVERS   = int(os.getenv("DC_NUM_SERVERS",    "100"))
NUM_RACKS     = int(os.getenv("DC_NUM_RACKS",      "10"))
NUM_UPS       = int(os.getenv("DC_NUM_UPS",        "4"))
NUM_INVERTERS = int(os.getenv("DC_NUM_INVERTERS",  "2"))


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------
def build_producer() -> Producer:
    conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "client.id":         "dc-energy-producer",
        "acks":              "all",
        "retries":           5,
        "linger.ms":         100,    # micro-batching for throughput
        "compression.type":  "lz4",
    }
    return Producer(conf)


def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")


def publish(producer: Producer, topic: str, records: list) -> int:
    """Serialize records to JSON and publish to Kafka topic."""
    sent = 0
    for rec in records:
        payload = json.dumps(asdict(rec), default=str).encode("utf-8")
        try:
            producer.produce(
                topic,
                value=payload,
                key=rec.__class__.__name__.encode(),
                callback=delivery_report,
            )
            sent += 1
        except KafkaException as e:
            logger.error(f"Kafka produce error: {e}")
    producer.poll(0)  # trigger delivery callbacks
    return sent


# ---------------------------------------------------------------------------
# Weather thread (runs every 15 min to avoid API hammering)
# ---------------------------------------------------------------------------
def weather_thread_fn(producer: Producer, stop_event: threading.Event):
    try:
        from data_generator.weather_api import WeatherClient
        client = WeatherClient()
    except ImportError:
        logger.warning("httpx not installed — weather thread disabled")
        return

    while not stop_event.is_set():
        reading = client.get_current()
        if reading:
            payload = json.dumps(asdict(reading), default=str).encode("utf-8")
            producer.produce(TOPIC_WEATHER, value=payload, callback=delivery_report)
            producer.flush()
            logger.info(f"[WEATHER] Published: {reading.temperature_c}°C, {reading.relative_humidity_pct}% RH")
        stop_event.wait(WEATHER_INTERVAL)


# ---------------------------------------------------------------------------
# Main producer loop
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Datacenter Energy Intelligence Platform — Kafka Producer")
    logger.info(f"  Broker : {BOOTSTRAP_SERVERS}")
    logger.info(f"  Servers: {NUM_SERVERS} | UPS: {NUM_UPS} | Inverters: {NUM_INVERTERS}")
    logger.info(f"  Interval: {INTERVAL_SECONDS}s")
    logger.info("=" * 60)

    producer    = build_producer()
    srv_sim     = ServerSimulator(num_servers=NUM_SERVERS, num_racks=NUM_RACKS)
    ups_sim     = UPSSimulator(num_ups=NUM_UPS)
    inv_sim     = InverterSimulator(num_inverters=NUM_INVERTERS)

    stop_event  = threading.Event()

    # Launch weather ingestion in a separate thread
    w_thread = threading.Thread(
        target=weather_thread_fn,
        args=(producer, stop_event),
        daemon=True,
        name="WeatherThread",
    )
    w_thread.start()

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received — flushing producer...")
        stop_event.set()
        producer.flush(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    tick = 0
    while True:
        tick += 1
        ts = datetime.now(timezone.utc)

        srv_records = srv_sim.generate_snapshot(ts)
        ups_records = ups_sim.generate_snapshot(ts)
        inv_records = inv_sim.generate_snapshot(ts)

        n_srv = publish(producer, TOPIC_SERVERS,   srv_records)
        n_ups = publish(producer, TOPIC_UPS,       ups_records)
        n_inv = publish(producer, TOPIC_INVERTERS, inv_records)
        producer.flush()

        # Count events
        anomalies = sum(1 for r in srv_records if r.is_anomaly)
        islanding = sum(1 for r in inv_records if r.islanding_detected)

        logger.info(
            f"[Tick {tick:05d}] {ts.strftime('%H:%M:%S')} | "
            f"Servers: {n_srv} | UPS: {n_ups} | Inverters: {n_inv} | "
            f"Anomalies: {anomalies} | Islanding: {islanding}"
        )

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
