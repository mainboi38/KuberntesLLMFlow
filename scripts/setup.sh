#!/bin/bash

set -e

echo "Setting up LLM Kubernetes Production Environment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }

# Create namespace
echo "Creating namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Install NGINX Ingress Controller
echo "Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Wait for ingress controller
echo "Waiting for ingress controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# Build Docker images for the host architecture
echo "Building Docker images for local architecture..."
ARCH=$(uname -m)
PLATFORM="linux/amd64"
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    PLATFORM="linux/arm64"
fi

docker buildx build --platform "$PLATFORM" -t llm-service:latest -f docker/llm-service/Dockerfile docker/llm-service/ --load
docker buildx build --platform "$PLATFORM" -t airflow:latest -f docker/airflow/Dockerfile docker/airflow/ --load

# Load images into kind/minikube if using local cluster
if [[ $(kubectl config current-context) == "kind-"* ]]; then
    echo "Loading images into kind cluster..."
    kind load docker-image llm-service:latest
    kind load docker-image airflow:latest
elif [[ $(kubectl config current-context) == "minikube" ]]; then
    echo "Loading images into minikube..."
    minikube image load llm-service:latest
    minikube image load airflow:latest
fi

# Create ConfigMaps for Airflow DAGs
echo "Creating Airflow DAGs ConfigMap..."
kubectl create configmap airflow-dags \
  --from-file=airflow/dags/ \
  --namespace=llm-production \
  --dry-run=client -o yaml | kubectl apply -f -

# Generate secrets
echo "Generating secrets..."
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
SECRET_KEY=$(openssl rand -hex 32)

# Create secrets file
cat > kubernetes/airflow/airflow-secrets-generated.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: airflow-secrets
  namespace: llm-production
type: Opaque
stringData:
  AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres-airflow:5432/airflow
  AIRFLOW__CELERY__BROKER_URL: redis://:@redis-airflow:6379/0
  AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres-airflow:5432/airflow
  AIRFLOW__CORE__FERNET_KEY: "$FERNET_KEY"
  AIRFLOW__WEBSERVER__SECRET_KEY: "$SECRET_KEY"
EOF

echo "Setup complete! Run ./scripts/deploy.sh to deploy the application."

