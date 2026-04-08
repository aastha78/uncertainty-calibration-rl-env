"""Inference script for the Uncertainty Calibration Environment."""

import os
import json
import re
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """/no_think
You are answering questions and rating your confidence.

For each question, respond ONLY with valid JSON:
{
  "answer": "your answer here",
  "confidence_1_5": 3,
  "uncertainty_type": "none"
}

Rules:
- confidence_1_5: integer 1-5 (1=no idea, 2=guess, 3=somewhat sure, 4=confident, 5=certain)
- uncertainty_type: one of "none", "partial", "full", "outdated", "contested"
- If you don't know, say so honestly with confidence 1
- NEVER rate 5 unless you are absolutely certain"""


def parse_llm_response(text: str) -> dict:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            p = json.loads(text[start:end])
            if "confidence_1_5" in p:
                p["confidence"] = max(0.0, min(1.0, (float(p["confidence_1_5"]) - 1) / 4.0))
            elif "confidence" not in p:
                p["confidence"] = 0.5
            p.setdefault("uncertainty_type", "none")
            return p
        except json.JSONDecodeError:
            pass
    return {"answer": text, "confidence": 0.5, "uncertainty_type": "none"}


def run_task(task_id: str):
    rewards = []
    steps_taken = 0
    success = False

    print(f"[START] task={task_id} env=uncertainty-calibration model={MODEL_NAME}", flush=True)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        obs = resp.json()

        while not obs.get("done", False):
            question = obs["question"]
            if not question:
                break

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                temperature=0.3,
                max_tokens=256,
            )

            llm_text = completion.choices[0].message.content
            parsed = parse_llm_response(llm_text)

            step_payload = {
                "answer": parsed.get("answer", "I don't know"),
                "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
                "uncertainty_type": parsed.get("uncertainty_type", "none"),
            }

            resp = requests.post(f"{ENV_URL}/step", json=step_payload)
            obs = resp.json()

            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            error = obs.get("error", None)
            rewards.append(reward)
            steps_taken += 1

            action_str = json.dumps(step_payload)
            print(
                f"[STEP] step={steps_taken} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error if error else 'null'}",
                flush=True,
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )

    return score


def main():
    tasks = ["task1_facts", "task2_partial", "task3_traps"]
    for task_id in tasks:
        run_task(task_id)


if __name__ == "__main__":
    main()
