import time
import uuid
import logging
import json
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import asyncio

# ----------------------
# Logging setup: write into logging.txt
# ----------------------
logging.basicConfig(
    filename="logging.txt",    # log file
    filemode="a",              # append on each run
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api_logger")

# ----------------------
# Prometheus metrics
# ----------------------
REQUEST_LATENCY = Histogram(
    "fastapi_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "endpoint", "http_status"]
)
REQUEST_COUNT = Counter(
    "fastapi_request_count_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"]
)
REQUEST_FAILURES = Counter(
    "fastapi_request_failures_total",
    "Total number of failed HTTP requests (status >= 400)",
    ["method", "endpoint", "http_status"]
)

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()
API_KEY = "a6f58e8a1c3e61d6f02b6c7b56d7b255831bb7fa1f2447bd8f1ac9283f083d6b"

class QueryRequest(BaseModel):
    premises_NL: List[str] = Field(..., alias="premises-NL")
    questions: List[str]

    model_config = {"populate_by_name": True, "from_attributes": True}

class QueryResponse(BaseModel):
    answers: List[str]
    idx: List[List[int]]
    explanation: List[str]

# $$$$$$$$$$$$$$$$$$$$$$
# DUMMY FUCNTION
# ----------------------
# ----------------------
# Initialize the model and resources (called during startup)
# ----------------------
def init_model():
    # Here you can initialize and load your model, for example:
    # model = load_model("path/to/your/model")
    # For example purposes, let's just simulate this with a dummy return:
    model = "initialized_model"  # Replace with your actual model initialization
    logger.info("Model initialized")
    return model

# ----------------------
# Optimize reasoning function
# ----------------------
def dummy_reasoning(premises: List[str], questions: List[str], model):
    answers, idx, explanations = [], [], []
    for question in questions:
        if "not" in question.lower():
            answers.append("No")
            idx.append([1])
            explanations.append("Premise 1 indicates negation, so the answer is No.")
        else:
            answers.append("Yes")
            idx.append([1, 2])
            explanations.append("Premises 1 and 2 support the positive answer Yes.")
    return answers, idx, explanations
# $$$$$$$$$$$$$$$$$$$$$$

# Load the model, modules, etc during startup
model = init_model()

# ----------------------
# Asynchronously save response to JSON
# ----------------------
async def save_to_json_async(question_id, premises_NL, questions, answers, idx, explanations):
    # Create questions directory if it doesn't exist
    if not os.path.exists("questions"):
        os.makedirs("questions")
    
    # Prepare the data to save, including premises-NL and questions
    data = {
        "premises-NL": premises_NL,
        "questions": questions,
        "answers": answers,
        "idx": idx,
        "explanations": explanations
    }
    
    # Save the data into a JSON file with the question_id
    filename = f"questions/{question_id}.json"
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    logger.info(f"Saved response to {filename}")

# ----------------------
# Middleware for metrics & logging
# ----------------------
@app.middleware("http")
async def metrics_and_logging(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        status_code = 500
        REQUEST_COUNT.labels(request.method, request.url.path, status_code).inc()
        REQUEST_FAILURES.labels(request.method, request.url.path, status_code).inc()
        logger.exception(f"{request.method} {request.url.path} -> {status_code} (unhandled error)")
        raise
    latency = time.time() - start_time

    status_code = response.status_code
    REQUEST_LATENCY.labels(request.method, request.url.path, status_code).observe(latency)
    REQUEST_COUNT.labels(request.method, request.url.path, status_code).inc()
    if status_code >= 400:
        REQUEST_FAILURES.labels(request.method, request.url.path, status_code).inc()

    logger.info(f"{request.method} {request.url.path} -> {status_code} in {latency:.3f}s")
    return response

# ----------------------
# /query endpoint with per-request logging
# ----------------------
@app.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    req_id = uuid.uuid4().hex
    start_time = time.time()
    logger.info(f"[{req_id}] START /query – received {len(body.questions)} questions")

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY}":
        duration = time.time() - start_time
        logger.warning(f"[{req_id}] UNAUTHORIZED – took {duration:.3f}s")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # ----------------------
        # Replace with actual function (a whole flow)
        answers, idx, explanations = dummy_reasoning(body.premises_NL, body.questions, model)
        # ----------------------
        duration = time.time() - start_time
        logger.info(f"[{req_id}] SUCCESS /query – processed in {duration:.3f}s")
        
        # Save response asynchronously after response is returned
        asyncio.create_task(save_to_json_async(req_id, body.premises_NL, body.questions, answers, idx, explanations))
        
        return QueryResponse(answers=answers, idx=idx, explanation=explanations)
    except Exception as exc:
        duration = time.time() - start_time
        logger.error(f"[{req_id}] ERROR /query – failed in {duration:.3f}s: {exc}")
        raise

# ----------------------
# /metrics endpoint for Prometheus
# ----------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

# ----------------------
# Run the app
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=43210)
