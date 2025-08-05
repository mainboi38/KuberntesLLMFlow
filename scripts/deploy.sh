#!/bin/bash

set -e

echo "Deploying LLM Production Environment..."

# Deploy Redis
echo "Deploying Redis..."
kubectl apply -f kubernetes/redis.yaml

# Deploy LLM Service components
echo "Deploying LLM Service..."
kubectl apply -f kubernetes/llm-service/configmap.yaml
kubectl apply -f kubernetes/llm-service/pvc.yaml
kubectl apply -f kubernetes/llm-service/deployment.yaml
kubectl apply -f kubernetes/llm-service/service.yaml
kubectl apply -f kubernetes/llm-service/hpa.yaml

# Deploy Airflow components
echo "Deploying Airflow..."
kubectl apply -f kubernetes/airflow/rbac.yaml
kubectl apply -f kubernetes/airflow/airflow-postgres.yaml
kubectl apply -f kubernetes/airflow/airflow-redis.yaml
kubectl apply -f kubernetes/airflow/airflow-configmap.yaml
kubectl apply -f kubernetes/airflow/airflow-secrets-generated.yaml
kubectl apply -f kubernetes/airflow/airflow-webserver.yaml
kubectl apply -f kubernetes/airflow/airflow-scheduler.yaml
kubectl apply -f kubernetes/airflow/airflow-worker.yaml

# Deploy Monitoring
echo "Deploying Monitoring Stack..."
kubectl apply -f kubernetes/monitoring/prometheus.yaml
kubectl apply -f kubernetes/monitoring/grafana.yaml

# Deploy Ingress
echo "Deploying Ingress..."
kubectl apply -f kubernetes/ingress.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n llm-production

# Update /etc/hosts for local development
echo "Updating /etc/hosts (may require sudo password)..."
echo "
# LLM Production Environment
127.0.0.1 llm.local
127.0.0.1 airflow.local
127.0.0.1 grafana.local
127.0.0.1 prometheus.local
" | sudo tee -a /etc/hosts

# Print access information
echo "
================================================================================
Deployment Complete!

Access your services at:
- LLM Service: http://llm.local
- Airflow UI: http://airflow.local (admin/admin)
- Grafana: http://grafana.local (admin/admin)
- Prometheus: http://prometheus.local

Port forwarding (if needed):
kubectl port-forward -n llm-production svc/llm-service 8000:80
kubectl port-forward -n llm-production svc/airflow-webserver 8080:8080
kubectl port-forward -n llm-production svc/grafana 3000:3000

Check pod status:
kubectl get pods -n llm-production

View logs:
kubectl logs -n llm-production -l app=llm-service
kubectl logs -n llm-production -l app=airflow-scheduler
================================================================================
"