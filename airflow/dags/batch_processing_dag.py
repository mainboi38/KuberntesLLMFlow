from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.sensors.python import PythonSensor
import requests
import json
import logging
import time

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'llm_batch_processing',
    default_args=default_args,
    description='Batch processing pipeline for LLM tasks',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    tags=['llm', 'batch', 'ml-pipeline']
)

def prepare_batch_data(**context):
    """Prepare data for batch processing"""
    # In production, this would fetch from a data warehouse or queue
    batch_data = {
        "customer_queries": [
            "What are your business hours?",
            "How do I reset my password?",
            "What payment methods do you accept?",
            "Can I track my order?",
            "What is your return policy?"
        ],
        "email_subjects": [
            "Welcome to our service",
            "Your order has been shipped",
            "Important account update",
            "Special offer just for you",
            "Thank you for your feedback"
        ],
        "product_descriptions": [
            {"name": "Smart Watch Pro", "features": ["GPS", "Heart Rate", "Water Resistant"]},
            {"name": "Wireless Earbuds", "features": ["Noise Cancelling", "30hr Battery", "Touch Controls"]},
            {"name": "Fitness Tracker", "features": ["Sleep Monitoring", "Step Counter", "Calorie Tracking"]}
        ]
    }
    
    context['task_instance'].xcom_push(key='batch_data', value=batch_data)
    return batch_data

def submit_batch_jobs(**context):
    """Submit multiple batch jobs to LLM service"""
    batch_data = context['task_instance'].xcom_pull(key='batch_data')
    llm_service_url = "http://llm-service.llm-production.svc.cluster.local"
    
    batch_jobs = []
    
    # Customer query responses
    query_prompts = [
        f"Generate a helpful and professional customer service response to: '{query}'"
        for query in batch_data['customer_queries']
    ]
    
    response = requests.post(
        f"{llm_service_url}/batch-generate",
        json={
            "prompts": query_prompts,
            "max_length": 256,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        batch_id = response.json()['batch_id']
        batch_jobs.append({
            "batch_id": batch_id,
            "type": "customer_queries",
            "count": len(query_prompts)
        })
        logging.info(f"Submitted customer query batch: {batch_id}")
    
    # Email content generation
    email_prompts = [
        f"Write a professional email with subject '{subject}'. Include greeting, main content, and call-to-action."
        for subject in batch_data['email_subjects']
    ]
    
    response = requests.post(
        f"{llm_service_url}/batch-generate",
        json={
            "prompts": email_prompts,
            "max_length": 512,
            "temperature": 0.8
        }
    )
    
    if response.status_code == 200:
        batch_id = response.json()['batch_id']
        batch_jobs.append({
            "batch_id": batch_id,
            "type": "emails",
            "count": len(email_prompts)
        })
        logging.info(f"Submitted email batch: {batch_id}")
    
    # Product descriptions
    product_prompts = [
        f"Write an engaging product description for {prod['name']} with features: {', '.join(prod['features'])}"
        for prod in batch_data['product_descriptions']
    ]
    
    response = requests.post(
        f"{llm_service_url}/batch-generate",
        json={
            "prompts": product_prompts,
            "max_length": 384,
            "temperature": 0.9
        }
    )
    
    if response.status_code == 200:
        batch_id = response.json()['batch_id']
        batch_jobs.append({
            "batch_id": batch_id,
            "type": "products",
            "count": len(product_prompts)
        })
        logging.info(f"Submitted product description batch: {batch_id}")
    
    context['task_instance'].xcom_push(key='batch_jobs', value=batch_jobs)
    return batch_jobs

def check_batch_completion(**context):
    """Check if all batch jobs are completed"""
    batch_jobs = context['task_instance'].xcom_pull(key='batch_jobs')
    llm_service_url = "http://llm-service.llm-production.svc.cluster.local"
    
    all_completed = True
    
    for job in batch_jobs:
        response = requests.get(f"{llm_service_url}/batch-status/{job['batch_id']}")
        if response.status_code == 200:
            status = response.json()['status']
            if status != 'completed':
                all_completed = False
                logging.info(f"Batch {job['batch_id']} status: {status}")
        else:
            all_completed = False
            
    return all_completed

def process_batch_results(**context):
    """Process and store batch results"""
    batch_jobs = context['task_instance'].xcom_pull(key='batch_jobs')
    llm_service_url = "http://llm-service.llm-production.svc.cluster.local"
    
    processed_results = {
        "customer_queries": [],
        "emails": [],
        "products": []
    }
    
    for job in batch_jobs:
        response = requests.get(f"{llm_service_url}/batch-status/{job['batch_id']}")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'completed' and 'results' in data:
                processed_results[job['type']] = data['results']
                logging.info(f"Processed {len(data['results'])} results for {job['type']}")
    
    # In production, store results in database or data warehouse
    total_processed = sum(len(results) for results in processed_results.values())
    logging.info(f"Total processed items: {total_processed}")
    
    context['task_instance'].xcom_push(key='processed_results', value=processed_results)
    return total_processed

def generate_analytics_report(**context):
    """Generate analytics report from batch processing"""
    processed_results = context['task_instance'].xcom_pull(key='processed_results')
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_items_processed": sum(len(results) for results in processed_results.values()),
            "average_generation_time": 0,
            "total_tokens_generated": 0
        },
        "by_category": {}
    }
    
    for category, results in processed_results.items():
        if results:
            total_time = sum(r.get('generation_time', 0) for r in results)
            total_tokens = sum(r.get('tokens_generated', 0) for r in results)
            
            report["by_category"][category] = {
                "count": len(results),
                "average_time": total_time / len(results) if results else 0,
                "total_tokens": total_tokens
            }
            
            report["summary"]["average_generation_time"] += total_time
            report["summary"]["total_tokens_generated"] += total_tokens
    
    if report["summary"]["total_items_processed"] > 0:
        report["summary"]["average_generation_time"] /= report["summary"]["total_items_processed"]
    
    logging.info(f"Analytics report generated: {json.dumps(report, indent=2)}")
    return report

# Define tasks
prepare_data_task = PythonOperator(
    task_id='prepare_batch_data',
    python_callable=prepare_batch_data,
    dag=dag
)

submit_jobs_task = PythonOperator(
    task_id='submit_batch_jobs',
    python_callable=submit_batch_jobs,
    dag=dag
)

# Sensor to check batch completion
batch_completion_sensor = PythonSensor(
    task_id='wait_for_batch_completion',
    python_callable=check_batch_completion,
    poke_interval=30,  # Check every 30 seconds
    timeout=1800,  # Timeout after 30 minutes
    mode='poke',
    dag=dag
)

process_results_task = PythonOperator(
    task_id='process_batch_results',
    python_callable=process_batch_results,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_analytics_report',
    python_callable=generate_analytics_report,
    dag=dag
)

# Advanced Kubernetes job for model fine-tuning
model_optimization_task = KubernetesPodOperator(
    task_id='model_optimization',
    name='llm-model-optimizer',
    namespace='llm-production',
    image='llm-service:latest',
    cmds=['python', '-c'],
    arguments=['''
import os
import json
import logging

# Simulate model optimization task
logging.info("Starting model optimization...")

# In production, this would:
# 1. Load performance metrics from recent batches
# 2. Analyze model performance
# 3. Potentially trigger fine-tuning or parameter adjustment
# 4. Update model configuration

optimization_results = {
    "timestamp": "2024-01-01T00:00:00Z",
    "metrics_analyzed": 1000,
    "recommendations": [
        "Increase temperature for creative tasks",
        "Reduce max_length for customer queries",
        "Enable 4-bit quantization for better performance"
    ],
    "status": "completed"
}

print(json.dumps(optimization_results, indent=2))
    '''],
    dag=dag,
    get_logs=True,
    is_delete_operator_pod=True,
    resources={
        'request_memory': '4Gi',
        'request_cpu': '2',
        'limit_memory': '8Gi',
        'limit_cpu': '4'
    }
)