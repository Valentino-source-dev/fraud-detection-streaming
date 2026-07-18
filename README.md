# 🛡️ Real-Time Fraud Detection Streaming Pipeline & MLOps

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg?style=flat-square)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg?style=flat-square&logo=docker)](https://www.docker.com/)
[![Redpanda](https://img.shields.io/badge/message%20broker-Redpanda-red.svg?style=flat-square&logo=redpanda)](https://redpanda.com/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-orange.svg?style=flat-square&logo=mlflow)](https://mlflow.org/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue.svg?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange.svg?style=flat-square&logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/observability-Grafana-orange.svg?style=flat-square&logo=grafana)](https://grafana.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An **End-to-End** ultra-low latency architecture for credit card fraud detection. This project bridges the worlds of **Real-Time Data Engineering** and **MLOps** to demonstrate how to train, promote, serve, and monitor a machine learning model in production with zero training-serving skew.

---

## 📐 System Architecture

The infrastructure is fully containerized and structured as a continuous feedback loop:

```mermaid
flowchart TB
    subgraph "Data Generation"
        CSV[(creditcard.csv)] -->|Replay at 10x| GEN[Generator Container]
    end

    subgraph "Streaming & Message Queue"
        GEN -->|Publish JSON events| RP{Redpanda / Kafka}
    end

    subgraph "Real-Time Inference (MLOps)"
        RP -->|Consume events| CONS[Stream Consumer]
        MLFLOW[(MLflow Registry)] -->|Load @production Model| CONS
        CONS -->|Engineered Features| XGB[XGBoost Model]
        XGB -->|Predict Fraud Score| CONS
    end

    subgraph "Storage & Observability"
        CONS -->|Batch insert predictions| DB[(PostgreSQL)]
        CONS -->|Expose metrics| PROM[Prometheus]
        PROM -->|Visualize live data| GRAF[Grafana Dashboard]
    end

    style RP fill:#f96,stroke:#333,stroke-width:2px,color:#000
    style MLFLOW fill:#ffb366,stroke:#333,stroke-width:2px,color:#000
    style XGB fill:#66ccff,stroke:#333,stroke-width:2px,color:#000
    style GRAF fill:#99ff99,stroke:#333,stroke-width:2px,color:#000
```

### 🔁 MLOps Lifecycle

In addition to data flow, the system enforces a strict MLOps lifecycle dividing offline experimentation, governance, and online inference:

```mermaid
flowchart TD
    subgraph "1. Experimentation & Training (Offline)"
        A[Raw CSV Data] --> B[Optuna Hyperparameter Tuning]
        B --> C[Model Evaluation & Metrics]
        C -->|Log Run & Artifacts| D[MLflow Tracking Server]
    end

    subgraph "2. Model Governance & Registry"
        D -->|Register Best Candidate| E[MLflow Model Registry]
        E -->|Promote to Production| F(Assign Alias: @production)
    end

    subgraph "3. Real-Time Serving (Online)"
        F -->|Hot-Load @production| G[Stream Consumer]
        H[Live Transaction Stream] --> G
        G -->|XGBoost Prediction| I[Database / Dashboard]
    end

    subgraph "4. Feedback & Monitoring Loop"
        G -->|Export Latency & Score Metrics| J[Prometheus / Grafana]
        J -->|Performance Drift Alert| K[Retrigger Training Pipeline]
        K -.->|Automated Run| B
    end

    style D fill:#ffb366,stroke:#333,stroke-width:1px,color:#000
    style E fill:#ffb366,stroke:#333,stroke-width:2px,color:#000
    style F fill:#ff9999,stroke:#333,stroke-width:2px,color:#000
    style G fill:#66ccff,stroke:#333,stroke-width:2px,color:#000
    style J fill:#99ff99,stroke:#333,stroke-width:1px,color:#000
```

---

## ✨ Key Features

* **🚀 Zero Training-Serving Skew**: Calculation of historical sliding window features (*rolling amount z-score*, *time gap*, *rolling count*) synchronized to simulated event time to ensure exact equivalence between offline training and live stream.
* **🧠 Advanced Optuna Training**: XGBoost training pipeline with automated hyperparameter search (50 trials, 3-fold CV) specifically optimized for **Area Under Precision-Recall Curve (AUC-PR)** on highly imbalanced data (0.17% frauds).
* **🗃️ Model Registry with Aliases**: Utilization of the modern MLflow `@production` system to promote and hot-serve models without downtime or consumer restarts.
* **📉 High-Performance Batch Writes**: The consumer accumulates predictions in memory and micro-batches them to PostgreSQL (every 50 records or 5 seconds) to maximize database throughput.
* **📊 Enterprise-Grade Observability**: Real-time monitoring of throughput, inference latency (averaging **~4ms**), consumer lag, and anomaly score via Prometheus and Grafana.
* **🛡️ 100% Test Coverage**: 24 unit and equivalence tests run continuously via CI.

---

## 🛠️ Tech Stack

* **Data Stream**: Redpanda (Kafka v3 compatible) + Redpanda Console
* **Machine Learning**: XGBoost, Scikit-learn, Optuna (Tuning), SHAP (Interpretability)
* **MLOps**: MLflow 2.15 (Tracking & Registry)
* **Database**: PostgreSQL (Historical sink)
* **Metrics**: Prometheus & Grafana

---

## ⏱️ Quick Start

### 1. Prerequisites
Make sure you have installed:
* Docker & Docker Compose
* Python 3.11 or 3.12 (for local training)

### 2. Dataset Download
Download the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle and place the `creditcard.csv` file in the `data/` folder:
```bash
# Expected structure:
# data/creditcard.csv
```

### 3. Training & Model Registration
Start the initial databases, install local requirements, and launch the training (Optuna optimization will take about 5-10 minutes depending on the CPU):
```bash
# Start Postgres and MLflow
docker compose up -d postgres mlflow

# Install local requirements
pip install -r training/requirements.txt

# Start training (Optuna -> Model Registration)
cd training && python3 train.py
```
*The winning model will be saved on MLflow and automatically promoted with the `@production` alias.*

### 4. Start Streaming Pipeline
```bash
# Return to the main folder
cd ..

# Copy the env config file
cp .env.example .env

# Start the entire pipeline
make up
```

### 5. Explore Live Dashboards
* **Grafana (Charts & Metrics)**: [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin`)
* **MLflow UI (Model Tracking)**: [http://localhost:5000](http://localhost:5000)
* **Redpanda Console (Kafka Messages)**: [http://localhost:8080](http://localhost:8080)

---

## 📂 Project Structure

```text
├── generator/          # Kafka producer — reads CSV and simulates events
├── consumer/           # Kafka consumer — feature engineering + XGBoost inference + DB writing
├── training/           # Local ML Pipeline (train.py, evaluate.py, baseline.py)
├── monitoring/         # Prometheus configs & Grafana JSON dashboards
├── db/                 # PostgreSQL initialization SQL scripts
├── tests/              # Pytest unit and equivalence tests
└── docker-compose.yml  # Infrastructure definition (8 containers)
```

---

## 📜 Useful Commands (Makefile)

The project includes a `Makefile` to simplify daily management:
```bash
make up         # Start the entire pipeline in background
make down       # Stop and remove all containers
make logs       # View logs of all services in real time
make logs-consumer # Specifically follow the inference consumer logs
make test       # Run all 24 local unit tests
make clean      # Remove all containers, cache, and Docker volumes (hard reset)
make urls       # Print the list of all active dashboard URLs
```

---

## 🎓 Academic Contribution

This project is ideal as a foundation or case study for theses in **Data Science, Cloud Computing, and Software Engineering**. It demonstrates the practical implementation of:
1. **Latency and Efficiency**: Measuring inference metrics and database buffering.
2. **Imbalanced Classes**: Handling real-world datasets with <0.2% fraud using gradient weighting (`scale_pos_weight`) and Area Under Precision-Recall curve.
3. **Software Robustness**: Service isolation via containers, automated testing in CI, and resilience to single-component crashes.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
