"""FastAPI server for the Uncertainty Calibration Environment."""

import os
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from environment import UncertaintyEnvironment

app = FastAPI(
    title="Uncertainty Calibration Environment",
    description="An OpenEnv RL environment for training LLMs to express calibrated confidence.",
    version="0.1.0",
)

env = UncertaintyEnvironment()


# --- Request/Response schemas ---

class ResetRequest(BaseModel):
    task_id: str = Field(default="task1_facts", description="task1_facts | task2_partial | task3_traps")

class StepRequest(BaseModel):
    answer: str = Field(description="The agent's answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    uncertainty_type: str = Field(default="none", description="none | partial | full | outdated | contested")


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "running", "environment": "uncertainty-calibration", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_id=req.task_id)
        return asdict(obs)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task_id}. Use: task1_facts, task2_partial, task3_traps")


@app.post("/step")
def step(req: StepRequest):
    try:
        from models import UncertaintyAction
        action = UncertaintyAction(
            answer=req.answer,
            confidence=req.confidence,
            uncertainty_type=req.uncertainty_type,
        )
        obs = env.step(action)
        return asdict(obs)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return asdict(env.state)


@app.get("/tasks")
def tasks():
    return env.list_tasks()


@app.get("/calibration_curve")
def calibration_curve():
    """Return calibration data for visualization."""
    state = env.state
    if not state.episode_id:
        return {"error": "No episode running. Call /reset first."}
    return {
        "confidence_history": env._confidence_history,
        "accuracy_history": env._accuracy_history,
        "calibration_errors": env._calibration_errors,
        "avg_calibration_error": state.avg_calibration_error,
        "questions_correct": state.questions_correct,
        "questions_total": state.questions_total,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
