# ⚡ Datacenter Energy Intelligence Platform

> End-to-end data engineering platform for real-time datacenter telemetry, ML-powered anomaly detection, and Grid-Forming vs Grid-Following inverter analysis.

**Master's Thesis Project — Electrical Engineering (Power Systems)**  
Universidade Federal de Santa Catarina · Florianópolis, SC, Brazil

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Models](#ml-models)
- [Local Stack](#local-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Service URLs](#service-urls)
- [Key Results](#key-results)
- [Academic Context](#academic-context)
- [License](#license)

---
The dashboard can be viewed here: https://datacenter-energy-platform-nzqihcuimx7beqlqqfjhmt.streamlit.app/#a416fc4c
---
## Overview

This platform bridges **data engineering** and **power systems research** by building a production-grade observability stack for datacenter microgrid analysis. It simulates, streams, processes, and analyzes telemetry from 100 servers, 4 UPS units, and 4 inverters operating in both Grid-Following (GFL) and Grid-Forming (GFM) control modes.

**Core research question:** Can ML models reliably classify GFL vs GFM inverter operating modes in real time, and what telemetry features are most discriminative during islanding events?

### Key Capabilities

- **Real-time streaming** — Kafka pipeline ingesting 100+ records every 5 seconds
- **Medallion data lake** — Bronze → Silver → Gold architecture on MinIO/S3
- **Three ML models** — Anomaly detection, PUE forecasting (LSTM), GFL/GFM classifier
- **Full observability** — Airflow DAGs, MLflow experiment tracking, Grafana dashboards
- **Cloud-ready** — Terraform provisions equivalent AWS infrastructure (S3, Glue, Athena)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  ServerSimulator · UPSSimulator · InverterSimulator · Weather   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ 5-second intervals
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMING LAYER                             │
│         Apache Kafka  (4 topics)                                │
│  dc.telemetry.servers · dc.telemetry.ups                        │
│  dc.telemetry.inverters · dc.telemetry.weather                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAKE (MinIO / S3)                        │
│                                                                 │
│  🥉 Bronze   Raw Parquet, partitioned by date, 30-day TTL       │
│  🥈 Silver   Schema-validated, enriched, versioned              │
│  🥇 Gold     Aggregated KPIs, 15-min windows → PostgreSQL       │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Airflow DAG (every 15 min)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ML & ANALYTICS LAYER                          │
│                                                                 │
│  Anomaly Detection  →  Random Forest (F1=0.83, AUC=0.93)       │
│  PUE Forecasting    →  LSTM (MAE=0.000126, MAPE=0.013%)        │
│  GFL/GFM Classifier →  Random Forest (AUC=0.98, CV F1=0.69)   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│           Streamlit Dashboard · Grafana · MLflow UI             │
└─────────────────────────────────────────────────────────────────┘
```

---

## ML Models

| Model | Algorithm | Target | Key Metrics |
|---|---|---|---|
| Anomaly Detection | Random Forest + Isolation Forest | Server faults (thermal, power, zombie) | AUC=0.93 · F1=0.83 · Threshold=0.34 |
| PUE Forecasting | LSTM (2-layer, hidden=64) | Datacenter PUE, 1-hour horizon | MAE=0.000126 · MAPE=0.013% |
| GFL/GFM Classifier | Random Forest (multiclass) | GFL / GFM / Transitioning | AUC=0.98 · CV F1=0.69 |

### Model Design Decisions

**Anomaly Detection — Why Random Forest over Isolation Forest?**  
Both were evaluated. The Isolation Forest achieved ROC-AUC=0.84 but F1=0.12 on subtle faults. The supervised Random Forest reached AUC=0.93 and F1=0.83 by leveraging the available ground-truth labels. In production, Isolation Forest would complement the RF to detect novel fault patterns never seen in training.

**GFL/GFM Classifier — Why CV F1=0.69?**  
The lower macro F1 reflects the difficulty of classifying the `transitioning` state, which lasts only 3–8 ticks and shares features with both GFL and GFM. The GFL and GFM classes individually achieve F1 > 0.90. This is academically interesting: the transitioning state is precisely when the inverter is most vulnerable, and accurate detection of this window is the core contribution.

---

## Local Stack

The local Docker environment mirrors a production cloud stack:

| Local Service | Cloud Equivalent | Purpose |
|---|---|---|
| MinIO | AWS S3 | Data lake object storage |
| Kafka | AWS Kinesis | Real-time telemetry streaming |
| PostgreSQL | AWS RDS | Gold layer KPI storage |
| Airflow | AWS MWAA | Pipeline orchestration |
| MLflow | AWS SageMaker Experiments | ML experiment tracking |
| Grafana | AWS CloudWatch | Operational dashboards |
| Streamlit | AWS EC2 / ECS | Business intelligence dashboard |

---

## Quick Start

### Prerequisites

- Docker Desktop ≥ 4.x
- Python 3.11 (Anaconda recommended)
- Git

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/datacenter-energy-platform.git
cd datacenter-energy-platform

conda create -n coding python=3.11 -y
conda activate coding
pip install confluent-kafka pandas numpy scipy loguru python-dotenv \
            jupyterlab plotly matplotlib seaborn scikit-learn xgboost \
            torch mlflow joblib
```

### 2. Start infrastructure

```bash
docker compose up -d
```

Wait ~2 minutes, then verify all services are running:

```bash
docker compose ps
```

Expected: 8 containers with status `Up` (airflow-init will show `Exited (0)` — this is correct).

### 3. Start data streaming

```bash
python -m ingestion.kafka_producer
```

Open [Kafka UI](http://localhost:8090) to see the 4 topics being populated in real time.

### 4. Run notebooks

```bash
jupyter lab
```

Run notebooks in order:
1. `notebooks/01_eda_datacenter.ipynb` — Exploratory Data Analysis
2. `notebooks/02_anomaly_detection.ipynb` — Anomaly Detection (RF + IF)
3. `notebooks/03_gfm_gfl_classifier.ipynb` — Inverter Mode Classification
4. `notebooks/04_pue_forecasting.ipynb` — LSTM PUE Forecasting

### 5. Run tests

```bash
pytest tests/test_simulators.py -v
```

---

## Project Structure

```
datacenter-energy-platform/
│
├── data_generator/
│   ├── server_simulator.py       # 100-server telemetry with fault injection
│   ├── ups_inverter_simulator.py # UPS + GFL/GFM inverter dynamics
│   └── weather_api.py            # Open-Meteo API (Florianópolis, SC)
│
├── ingestion/
│   └── kafka_producer.py         # Streams all simulators to Kafka
│
├── airflow_dags/
│   └── datacenter_pipeline.py    # Bronze→Silver→Gold ETL + drift detection
│
├── ml/
│   ├── anomaly_model.pkl         # Trained anomaly detector
│   ├── gfm_classifier.pkl        # Trained GFL/GFM classifier
│   ├── pue_lstm_best.pt          # Trained LSTM weights (PyTorch)
│   └── *.json                    # Feature configs
│
├── notebooks/
│   ├── 01_eda_datacenter.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_gfm_gfl_classifier.ipynb
│   └── 04_pue_forecasting.ipynb
│
├── dashboard/
│   └── app.py                    # Streamlit multi-page dashboard
│
├── infra/
│   ├── main.tf                   # AWS S3, Glue, Athena, IAM, CloudWatch
│   └── variables.tf
│
├── tests/
│   └── test_simulators.py        # 15 unit tests (pytest)
│
├── docker/
│   ├── postgres-init.sh          # Creates airflow/mlflow/datacenter_gold DBs
│   └── Dockerfile.streamlit
│
├── docker-compose.yml
├── requirements.txt
├── .env
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml      # GitHub Actions: lint + pytest + smoke tests
```

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | — |
| Airflow | http://localhost:8081 | admin / admin123 |
| MLflow | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Kafka UI | http://localhost:8090 | — |
| Grafana | http://localhost:3000 | admin / admin123 |
| PostgreSQL | localhost:5432 | admin / admin123 |
| Kafka Broker | localhost:29092 | — |

---

## Key Results

### Simulator Realism

| Parameter | Value | Basis |
|---|---|---|
| Server idle power | 45–80 W | ASHRAE thermal guidelines |
| Server TDP range | 150–350 W | Intel/AMD server CPUs |
| Diurnal peak | 14:00 UTC | Typical business hours pattern |
| PUE target | ~1.002 | Best-in-class hyperscaler |
| Fault rate | 1–5% | Configurable per experiment |

### GFL vs GFM Inverter Dynamics

| Metric | GFL | GFM | Standard |
|---|---|---|---|
| Avg ROCOF (normal) | 0.054 Hz/s | 0.063 Hz/s | IEEE 1547: < 0.5 Hz/s |
| THD (normal) | ~2.5% | ~3.3% | IEEE 519: < 5% |
| ROCOF (islanding) | > 2 Hz/s | < 0.5 Hz/s | Key thesis finding |
| Islanding stability | ❌ Unstable | ✅ Stable | VSM virtual inertia |

**Key finding:** In normal grid-connected operation, GFL and GFM exhibit similar ROCOF and THD. The critical difference emerges during islanding events, where GFL loses PLL synchronization (ROCOF > 2 Hz/s) while GFM maintains stable frequency through virtual synchronous machine dynamics. This validates the thesis hypothesis that GFM inverters provide superior resilience for datacenter microgrids.

### Pipeline Performance

| Stage | Records/tick | Latency | Retention |
|---|---|---|---|
| Kafka ingestion | ~106 records | < 100 ms | 24 hours |
| Bronze write | ~106 records | < 2 s | 30 days |
| Silver transform | Batched 15-min | ~10 s | Versioned |
| Gold aggregation | 15-min KPIs | ~5 s | Permanent |

---

## Academic Context

This project supports research on **Grid-Forming inverter integration in datacenter microgrids**, contributing to the following open questions:

1. **Observability:** Can a data engineering stack provide sufficient telemetry resolution (5-second intervals) to capture GFL→GFM transition dynamics?

2. **Classification:** Can ML models reliably distinguish inverter operating modes using electrical telemetry features, enabling automated grid event detection?

3. **Correlation:** Is there a statistically significant correlation between IT workload patterns (CPU utilization, power draw) and microgrid stability metrics (ROCOF, frequency deviation)?

4. **Predictive maintenance:** Can LSTM-based PUE forecasting provide actionable 1-hour ahead predictions for energy optimization decisions?

### Related Work

- IEEE 1547-2018: *Standard for Interconnection and Interoperability of Distributed Energy Resources*
- IEEE 519-2022: *Recommended Practice for Harmonic Control in Electric Power Systems*
- ASHRAE TC 9.9: *Thermal Guidelines for Data Processing Environments*

---

## AWS Deployment

To deploy the cloud infrastructure:

```bash
cd infra
terraform init
terraform plan
terraform apply
```

This provisions:
- S3 buckets (bronze / silver / gold / mlflow-artifacts)
- AWS Glue Catalog + crawler (auto-discovers Silver schema)
- Athena workgroup (1 GB query limit)
- IAM roles (Glue + application roles)
- CloudWatch log group

**Estimated cost:** ~$15–30/month for a development workload.

---

## CI/CD

GitHub Actions runs on every push to `main`:

1. **Lint** — ruff + black formatting check
2. **Test** — pytest with coverage report
3. **Smoke test** — validates simulators produce valid JSON output
4. **Docker** — validates `docker-compose.yml` syntax

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the intersection of data engineering and power systems research.*  
*Florianópolis, SC, Brazil · 2026*
