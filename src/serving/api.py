"""
api.py
------
FastAPI REST server for the fine-tuned Code Review LLM.

Endpoints:
  POST /review        — generate a code review for a given snippet
  GET  /health        — liveness check
  GET  /model-info    — return loaded model metadata

Run (development):
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

Run (production):
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "./outputs/dpo_model")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are an expert software engineer performing a thorough code review. "
    "Provide specific, actionable feedback on correctness, style, performance, "
    "and security. Be concise but precise."
)

# ── Global model store ─────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    _state["pipe"] = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    _state["model_path"] = MODEL_PATH
    _state["load_time"]  = round(time.time() - start, 2)
    logger.info(f"Model ready in {_state['load_time']}s ✓")
    yield
    logger.info("Shutting down — freeing model.")
    del _state["pipe"]


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Code Review LLM API",
    description="Fine-tuned LLaMA 3.1 8B (QLoRA + DPO) for automated code review",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    code: str = Field(..., description="The code snippet to review", min_length=10)
    language: str = Field(default="python", description="Programming language")
    max_new_tokens: Optional[int] = Field(
        default=None, description="Override default max tokens (max 1024)", le=1024
    )

class ReviewResponse(BaseModel):
    review: str
    language: str
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    device: str
    load_time_s: float
    max_new_tokens: int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=DEVICE,
        model_loaded="pipe" in _state,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if "pipe" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_path=_state["model_path"],
        device=DEVICE,
        load_time_s=_state["load_time"],
        max_new_tokens=MAX_NEW_TOKENS,
    )


@app.post("/review", response_model=ReviewResponse)
def review_code(request: ReviewRequest):
    if "pipe" not in _state:
        raise HTTPException(status_code=503, detail="Model is still loading")

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Code to Review:\n```{request.language}\n{request.code}\n```\n\n"
        f"### Review:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    max_tokens = request.max_new_tokens or MAX_NEW_TOKENS

    t0 = time.perf_counter()
    try:
        output = _state["pipe"](prompt, max_new_tokens=max_tokens)[0]["generated_text"]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Extract assistant response only
    if "<|start_header_id|>assistant<|end_header_id|>" in output:
        review = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        review = review.replace("<|eot_id|>", "").strip()
    else:
        review = output[len(prompt):].strip()

    logger.info(f"Generated review in {latency_ms}ms ({len(review.split())} words)")

    return ReviewResponse(
        review=review,
        language=request.language,
        latency_ms=latency_ms,
        model=Path(MODEL_PATH).name,
    )
