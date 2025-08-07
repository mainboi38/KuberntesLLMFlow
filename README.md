# Kubernetes LLM Flow

Kubernetes LLM Flow is a reference implementation for running large language model (LLM) workloads on a Kubernetes cluster. It combines a FastAPI-based inference service, scheduled data pipelines, and monitoring tooling to showcase how generative AI systems can be operated in a cloud-native environment.

## Problem it Tackles

Deploying and maintaining LLMs in production requires more than a model: teams need repeatable workflows, scalable infrastructure, and observability. This project demonstrates how to:

- Build a containerized LLM service with caching and metrics.
- Orchestrate batch generation and optimization jobs with Apache Airflow.
- Scale components up and down using Kubernetes primitives.
- Track system health through Prometheus-compatible metrics and Grafana dashboards.

By providing a complete stack, it addresses the pain points of ad‑hoc scripts and single‑machine deployments and offers a path toward reliable, automated generative AI operations.

## How It Works

### LLM Service
The `docker/llm-service` directory contains a FastAPI application that loads a transformer model and exposes REST endpoints. It supports single and batched text generation, optional Redis caching, and Prometheus metrics for observability.

### Airflow Pipeline
The `airflow/dags` folder defines a daily DAG that prepares batch data, submits it to the LLM service, polls for completion, aggregates results, and even runs a KubernetesPodOperator job to simulate model optimization.

### Kubernetes Manifests
`kubernetes/` holds manifests for deploying the entire stack:
- `llm-service/` defines a `Deployment`, `Service`, `ConfigMap`, `PersistentVolumeClaim`, and `HorizontalPodAutoscaler` for the model server.
- `airflow/` contains manifests for the scheduler, workers, and supporting PostgreSQL/Redis pods.
- `monitoring/` provides a Grafana instance and an ingress that routes requests to the LLM service, Airflow UI, Grafana, and Prometheus.

### Supporting Scripts
Two helper scripts automate cluster setup and deployment:
- `scripts/setup.sh` installs the ingress controller, creates namespaces and ConfigMaps, and generates the secrets required by Airflow.
- `scripts/deploy.sh` applies all Kubernetes resources and prints access URLs.

## Advantages of Running GenAI on Kubernetes

- **Elastic scaling:** Horizontal Pod Autoscalers adjust replica counts based on CPU and memory usage so that the service can meet changing demand without manual intervention.
- **Resource isolation:** Pods declare resource requests/limits, ensuring LLM workloads coexist safely with other services.
- **Workflow automation:** Airflow operators and sensors coordinate complex generation pipelines, making it easy to schedule or chain jobs.
- **Portability:** The entire stack runs from containers and YAML files, enabling reproducible deployments across local clusters and cloud providers.
- **Observability:** Prometheus metrics and Grafana dashboards give insight into request latency, token generation, and overall system health.

## Getting Started

1. **Build images**
   ```bash
   docker build -t llm-service:latest docker/llm-service
   ```
   The deployment uses the official Apache Airflow image `apache/airflow:2.7.3-python3.10`, so no additional build step is required.
2. **Set up cluster prerequisites**
   ```bash
   ./scripts/setup.sh
   ```
3. **Deploy the stack**
   ```bash
   ./scripts/deploy.sh
   ```
4. **Access services**
   - LLM API: `http://llm.local`
   - Airflow UI: `http://airflow.local`
   - Grafana: `http://grafana.local`
   - Prometheus: `http://prometheus.local`

## Repository Structure

```
├── airflow/                 # Airflow DAGs
├── docker/                  # Dockerfiles for services
├── kubernetes/              # Kubernetes manifests
└── scripts/                 # Setup and deployment scripts
```

This repository can serve as a starting point for experimenting with generative AI workloads on Kubernetes or as a foundation for more advanced production setups.
