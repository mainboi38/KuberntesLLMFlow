from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn
import logging
import asyncio
from datetime import datetime
import redis
import json
import os
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests')
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'LLM request latency')
TOKEN_COUNT = Counter('llm_tokens_generated_total', 'Total tokens generated')

app = FastAPI(title="LLM Service", version="1.0.0")

# Redis connection for caching
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-7b-chat-hf')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))

# Global model and tokenizer
model = None
tokenizer = None
generation_pipeline = None

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_length: Optional[int] = Field(default=MAX_LENGTH, description="Maximum length of generated text")
    temperature: Optional[float] = Field(default=TEMPERATURE, description="Temperature for generation")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter")
    cache_enabled: Optional[bool] = Field(default=True, description="Enable Redis caching")
    session_id: Optional[str] = Field(default=None, description="Session ID for context tracking")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int
    generation_time: float
    cached: bool = False
    session_id: Optional[str] = None

class BatchRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = Field(default=MAX_LENGTH)
    temperature: Optional[float] = Field(default=TEMPERATURE)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@app.on_event("startup")
async def load_model():
    """Load the LLM model on startup"""
    global model, tokenizer, generation_pipeline
    
    logger.info(f"Loading model: {MODEL_NAME}")
    try:
        # For production, you'd want to use actual Llama 3.1 weights
        # This example uses a smaller model for demonstration
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Enable 8-bit quantization for efficiency
        )
        
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text based on the input prompt"""
    REQUEST_COUNT.inc()
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Check cache if enabled
    cache_key = f"llm:generation:{hash(request.prompt)}:{request.max_length}:{request.temperature}"
    if request.cache_enabled:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            result['cached'] = True
            return GenerationResponse(**result)
    
    try:
        with REQUEST_LATENCY.time():
            # Generate text
            outputs = generation_pipeline(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                num_return_sequences=1
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the generated text if it's included
            if generated_text.startswith(request.prompt):
                generated_text = generated_text[len(request.prompt):].strip()
            
            # Count tokens
            tokens = tokenizer.encode(generated_text)
            TOKEN_COUNT.inc(len(tokens))
            
            generation_time = time.time() - start_time
            
            response_data = {
                "generated_text": generated_text,
                "prompt": request.prompt,
                "tokens_generated": len(tokens),
                "generation_time": generation_time,
                "session_id": request.session_id
            }
            
            # Cache the result
            if request.cache_enabled:
                redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(response_data)
                )
            
            # Store session context if provided
            if request.session_id:
                session_key = f"llm:session:{request.session_id}"
                session_data = {
                    "prompt": request.prompt,
                    "response": generated_text,
                    "timestamp": datetime.utcnow().isoformat()
                }
                redis_client.lpush(session_key, json.dumps(session_data))
                redis_client.expire(session_key, 86400)  # 24 hour TTL
            
            return GenerationResponse(**response_data)
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-generate")
async def batch_generate(request: BatchRequest, background_tasks: BackgroundTasks):
    """Process multiple prompts in batch"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    batch_id = f"batch:{datetime.utcnow().timestamp()}"
    
    # Store batch status
    redis_client.hset(f"llm:batch:{batch_id}", "status", "processing")
    redis_client.hset(f"llm:batch:{batch_id}", "total", len(request.prompts))
    redis_client.hset(f"llm:batch:{batch_id}", "completed", 0)
    
    # Process in background
    background_tasks.add_task(process_batch, batch_id, request)
    
    return {"batch_id": batch_id, "status": "processing", "total_prompts": len(request.prompts)}

async def process_batch(batch_id: str, request: BatchRequest):
    """Process batch generation in background"""
    results = []
    
    for i, prompt in enumerate(request.prompts):
        try:
            gen_request = GenerationRequest(
                prompt=prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                cache_enabled=False
            )
            result = await generate_text(gen_request)
            results.append(result.dict())
            
            # Update progress
            redis_client.hset(f"llm:batch:{batch_id}", "completed", i + 1)
            
        except Exception as e:
            results.append({"error": str(e), "prompt": prompt})
    
    # Store results
    redis_client.hset(f"llm:batch:{batch_id}", "status", "completed")
    redis_client.hset(f"llm:batch:{batch_id}", "results", json.dumps(results))
    redis_client.expire(f"llm:batch:{batch_id}", 3600)  # 1 hour TTL

@app.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of batch processing"""
    batch_key = f"llm:batch:{batch_id}"
    
    if not redis_client.exists(batch_key):
        raise HTTPException(status_code=404, detail="Batch not found")
    
    status = redis_client.hget(batch_key, "status")
    total = int(redis_client.hget(batch_key, "total"))
    completed = int(redis_client.hget(batch_key, "completed"))
    
    response = {
        "batch_id": batch_id,
        "status": status,
        "total": total,
        "completed": completed,
        "progress": completed / total if total > 0 else 0
    }
    
    if status == "completed":
        results = redis_client.hget(batch_key, "results")
        response["results"] = json.loads(results) if results else []
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
