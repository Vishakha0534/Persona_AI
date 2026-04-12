"""Baseline inference script that runs a deterministic baseline agent
against the OpenEnv environment's 3 difficulty tasks.

It will use the OpenAI API if `OPENAI_API_KEY` is present; otherwise a simple
heuristic agent is used. The script prints reproducible scores per task.
"""
import os
import random
from env.openenv_env import OpenEnv
from env.models import Action

try:
    import openai
except Exception:
    openai = None


SEED = 1234


def heuristic_agent(obs_text: str, difficulty: str, phase: int = 0):
    text = obs_text.lower()
    if difficulty == "easy":
        # simple priority heuristics
        if any(w in text for w in ["not arrived", "two weeks", "never received", "missing"]):
            return Action(action_type="label", label="high")
        if any(w in text for w in ["change address", "update address"]):
            return Action(action_type="label", label="medium")
        return Action(action_type="label", label="low")

    if difficulty == "medium":
        # produce a short reply containing keywords heuristically
        reply = "Hello, thanks for contacting us. "
        if "refund" in text or "wrong item" in text:
            reply += "I'm sorry about the wrong item — we can issue a refund. "
        reply += "Please provide any SKU/reference and we'll follow up. Regards."
        return Action(action_type="reply", content=reply)

    # hard: multi-step agent
    if difficulty == "hard":
        if phase == 0:
            return Action(action_type="label", label="critical")
        if phase == 1:
            summary = "Multiple services return 500 errors; user reports deployment failure."
            return Action(action_type="summary", content=summary)
        return Action(action_type="recommend", content="Restart affected services, monitor, and rollback if needed.")


def openai_agent(obs_text: str, difficulty: str, phase: int = 0):
    key = os.environ.get("OPENAI_API_KEY")
    if not key or openai is None:
        return heuristic_agent(obs_text, difficulty, phase)
    openai.api_key = key
    prompt = f"Triaging email (difficulty={difficulty}, phase={phase}): {obs_text}\nRespond with a single JSON object with keys: action_type, label (optional), content (optional)."
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # best-effort parse
    import json

    try:
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return heuristic_agent(obs_text, difficulty, phase)


def run_baseline(use_openai: bool = False):
    random.seed(SEED)
    results = {}
    for difficulty in ("easy", "medium", "hard"):
        env = OpenEnv(difficulty=difficulty, seed=SEED)
        obs = env.reset()
        total_reward = 0.0
        done = False
        phase = env.state().get("phase", 0)
        steps = 0
        while not done and steps < 10:
            if use_openai:
                act = openai_agent(obs.text, difficulty, phase)
            else:
                act = heuristic_agent(obs.text, difficulty, phase)
            obs, reward, done, info = env.step(act)
            total_reward += reward.reward
            phase = env.state().get("phase", 0)
            steps += 1
        avg = total_reward / max(1, steps)
        results[difficulty] = {"total": total_reward, "steps": steps, "avg": avg}
    return results


if __name__ == "__main__":
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print("Running baseline agent (openai=%s)" % use_openai)
    res = run_baseline(use_openai=use_openai)
    for k, v in res.items():
        print(f"{k}: total_reward={v['total']:.4f}, steps={v['steps']}, avg={v['avg']:.4f}")
