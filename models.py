"""Type-safe Pydantic models for the Uncertainty Calibration Environment."""

from pydantic import BaseModel, Field
from typing import List


class UncertaintyAction(BaseModel):
    """What the agent sends each step."""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_type: str = "none"


class UncertaintyObservation(BaseModel):
    """What the environment returns after each step."""
    question: str
    question_category: str
    is_correct: bool
    expected_confidence_range: List[float]
    ground_truth: str
    knowledge_category: str
    reward: float
    done: bool
    step_number: int
    total_steps: int
    feedback: str
    confidence_history: List[float] = []
    accuracy_history: List[bool] = []


class UncertaintyState(BaseModel):
    """Episode metadata."""
    episode_id: str
    task_id: str
    step_count: int = 0
    total_steps: int = 0
    cumulative_reward: float = 0.0
    avg_calibration_error: float = 0.0
    questions_correct: int = 0
    questions_total: int = 0


class TaskConfig(BaseModel):
    """Configuration for a task."""
    task_id: str
    name: str
    description: str
    max_steps: int
    data_file: str
