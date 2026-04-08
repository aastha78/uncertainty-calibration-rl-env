"""Type-safe models for the Uncertainty Calibration Environment."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class UncertaintyAction:
    """What the agent sends each step."""
    answer: str                    # The agent's answer to the question
    confidence: float              # 0.0 to 1.0 — how confident the agent is
    uncertainty_type: str = "none" # none | partial | full | outdated | contested


@dataclass
class UncertaintyObservation:
    """What the environment returns after each step."""
    question: str                          # Current/next question
    question_category: str                 # Category hint: factual, temporal, contested, etc.
    is_correct: bool                       # Was the agent's answer correct?
    expected_confidence_range: List[float] # [low, high] — ideal confidence range (revealed after grading)
    ground_truth: str                      # Correct answer (revealed after grading)
    knowledge_category: str                # KNOWN / PARTIAL / UNKNOWN / OUTDATED / CONTESTED
    reward: float                          # 0.0 to 1.0
    done: bool                             # Is the episode over?
    step_number: int
    total_steps: int
    feedback: str                          # Human-readable feedback
    confidence_history: List[float] = field(default_factory=list)  # Past confidence scores
    accuracy_history: List[bool] = field(default_factory=list)     # Past correctness


@dataclass
class UncertaintyState:
    """Episode metadata."""
    episode_id: str
    task_id: str                   # task1_facts / task2_partial / task3_traps
    step_count: int = 0
    total_steps: int = 0
    cumulative_reward: float = 0.0
    avg_calibration_error: float = 0.0
    questions_correct: int = 0
    questions_total: int = 0


@dataclass
class TaskConfig:
    """Configuration for a task."""
    task_id: str
    name: str
    description: str
    max_steps: int
    data_file: str
