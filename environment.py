"""Core environment logic for the Uncertainty Calibration Environment."""

import json
import uuid
import random
from pathlib import Path
from typing import Optional

from models import (
    UncertaintyAction,
    UncertaintyObservation,
    UncertaintyState,
    TaskConfig,
)
from reward import compute_reward, generate_feedback

DATA_DIR = Path(__file__).parent / "data"

TASKS = {
    "task1_facts": TaskConfig(
        task_id="task1_facts",
        name="Clear-Cut Facts",
        description="Questions with clearly knowable or unknowable answers. Calibrate confidence for binary know/don't-know.",
        max_steps=15,
        data_file="task1_facts.json",
    ),
    "task2_partial": TaskConfig(
        task_id="task2_partial",
        name="Shades of Grey",
        description="Questions with partial, outdated, or contested knowledge. Express graduated uncertainty.",
        max_steps=15,
        data_file="task2_partial.json",
    ),
    "task3_traps": TaskConfig(
        task_id="task3_traps",
        name="Confidence Traps",
        description="Questions designed to trigger overconfidence — misconceptions, trick premises, and precision traps.",
        max_steps=15,
        data_file="task3_traps.json",
    ),
}


def _load_questions(data_file: str) -> list:
    with open(DATA_DIR / data_file) as f:
        return json.load(f)


def _check_answer(agent_answer: str, question_data: dict) -> bool:
    """Check if the agent's answer matches any accepted answer."""
    agent_lower = agent_answer.strip().lower()

    for accepted in question_data.get("accept_answers", []):
        acc_lower = accepted.lower().strip()
        if not acc_lower:
            continue
        # Substring match (either direction)
        if acc_lower in agent_lower or agent_lower in acc_lower:
            return True
        # Word overlap: >60% of accept words appear in response
        acc_words = set(acc_lower.split())
        ans_words = set(agent_lower.split())
        if acc_words and len(acc_words & ans_words) / len(acc_words) > 0.6:
            return True

    # For UNKNOWN questions, "I don't know" type answers are correct
    if question_data.get("knowledge_category") == "UNKNOWN":
        if any(x in agent_lower for x in ["i don't know", "i dont know", "unknown", "cannot", "unknowable", "not possible", "impossible"]):
            return True

    # Premise rejection: if model challenges a false premise in misconception questions
    cat = question_data.get("category", "")
    if cat in ("misconception", "misattribution", "health_trap", "tricky_science"):
        rejection_signals = ["myth", "false", "incorrect", "not true", "misconception",
                           "actually", "contrary", "no evidence", "debunked", "premise"]
        if any(w in agent_lower for w in rejection_signals):
            return True

    return False


class UncertaintyEnvironment:
    """Uncertainty Calibration Environment."""

    def __init__(self):
        self._state: Optional[UncertaintyState] = None
        self._questions: list = []
        self._current_q_idx: int = 0
        self._task: Optional[TaskConfig] = None
        self._calibration_errors: list = []
        self._confidence_history: list = []
        self._accuracy_history: list = []

    def reset(self, task_id: str = "task1_facts") -> UncertaintyObservation:
        """Start a new episode."""
        self._task = TASKS[task_id]
        all_questions = _load_questions(self._task.data_file)
        self._questions = random.sample(all_questions, min(self._task.max_steps, len(all_questions)))
        self._current_q_idx = 0
        self._calibration_errors = []
        self._confidence_history = []
        self._accuracy_history = []

        self._state = UncertaintyState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            total_steps=len(self._questions),
            cumulative_reward=0.0,
            avg_calibration_error=0.0,
            questions_correct=0,
            questions_total=0,
        )

        q = self._questions[0]
        return UncertaintyObservation(
            question=q["question"],
            question_category=q["category"],
            is_correct=False,
            expected_confidence_range=[0.0, 1.0],
            ground_truth="",
            knowledge_category="",
            reward=0.0,
            done=False,
            step_number=0,
            total_steps=len(self._questions),
            feedback=f"Task: {self._task.name}. {self._task.description}",
        )

    def step(self, action: UncertaintyAction) -> UncertaintyObservation:
        """Process agent's answer + confidence, return reward and next question."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        q = self._questions[self._current_q_idx]
        is_correct = _check_answer(action.answer, q)

        # Adaptive penalty: if model was overconfident+wrong recently, increase penalty
        recent_overconf = sum(1 for ce in self._calibration_errors[-3:] if ce > 0.5)
        overconf_multiplier = 1.0 + 0.2 * recent_overconf  # up to 1.6x penalty

        # Map task to difficulty
        difficulty_map = {"task1_facts": "easy", "task2_partial": "medium", "task3_traps": "hard"}
        task_difficulty = difficulty_map.get(self._task.task_id, "medium")

        reward_info = compute_reward(
            confidence=action.confidence,
            is_correct=is_correct,
            predicted_type=action.uncertainty_type,
            actual_type=q["knowledge_category"],
            overconf_multiplier=overconf_multiplier,
            task_difficulty=task_difficulty,
        )

        self._calibration_errors.append(reward_info["calibration_error"])
        self._confidence_history.append(action.confidence)
        self._accuracy_history.append(is_correct)

        # Meta-reward: bonus for improving calibration over the episode
        meta_bonus = 0.0
        if len(self._calibration_errors) >= 3:
            recent_ce = sum(self._calibration_errors[-3:]) / 3
            earlier_ce = sum(self._calibration_errors[:-3]) / max(len(self._calibration_errors) - 3, 1)
            if recent_ce < earlier_ce:
                meta_bonus = 0.05  # small bonus for improving
        reward_info["total_reward"] = min(1.0, reward_info["total_reward"] + meta_bonus)
        self._state.step_count += 1
        self._state.questions_total += 1
        self._state.cumulative_reward += reward_info["total_reward"]
        self._state.avg_calibration_error = sum(self._calibration_errors) / len(self._calibration_errors)
        if is_correct:
            self._state.questions_correct += 1

        self._current_q_idx += 1
        done = self._current_q_idx >= len(self._questions)

        feedback = generate_feedback(
            confidence=action.confidence,
            is_correct=is_correct,
            reward_breakdown=reward_info,
            knowledge_category=q["knowledge_category"],
        )

        next_question = ""
        next_category = ""
        if not done:
            nq = self._questions[self._current_q_idx]
            next_question = nq["question"]
            next_category = nq["category"]

        return UncertaintyObservation(
            question=next_question,
            question_category=next_category,
            is_correct=is_correct,
            expected_confidence_range=q["expected_confidence"],
            ground_truth=q["ground_truth"],
            knowledge_category=q["knowledge_category"],
            reward=reward_info["total_reward"],
            done=done,
            step_number=self._state.step_count,
            total_steps=len(self._questions),
            feedback=feedback,
            confidence_history=list(self._confidence_history),
            accuracy_history=list(self._accuracy_history),
        )

    @property
    def state(self) -> UncertaintyState:
        if self._state is None:
            return UncertaintyState(
                episode_id="", task_id="", step_count=0, total_steps=0,
            )
        return self._state

    def list_tasks(self) -> list:
        return [
            {"task_id": t.task_id, "name": t.name, "description": t.description}
            for t in TASKS.values()
        ]
