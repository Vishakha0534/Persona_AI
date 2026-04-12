---
title: PersonaAI (OpenEnv email triage)
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

## PersonaAI — OpenEnv: Email & Incident Triage

This repository now contains a complete OpenEnv-compliant environment that simulates real-world email/incident triage tasks (not a game or toy). It includes three graded tasks (easy → medium → hard), typed Pydantic models, deterministic graders, a baseline inference script, and deployment metadata.

Highlights
- Real-world task: customer support / incident triage workflows
- Full OpenEnv spec: typed `Observation`, `Action`, `Reward`, `reset()`, `step()`, `state()`
- Three tasks with deterministic graders: `easy`, `medium`, `hard` (partial credit)
- Baseline inference script using OpenAI (optional) or deterministic heuristic
- `openenv.yaml` with metadata for validation and deployment

Files added/changed
- `env/models.py` — Pydantic models for Observation, Action, Reward
- `env/tasks.py` — example dataset + graders (easy/medium/hard)
- `env/openenv_env.py` — OpenEnv environment implementing reset/step/state
- `baseline.py` — baseline agent script (uses OPENAI_API_KEY if present)
- `openenv.yaml` — environment metadata

Quickstart

1. Install deps
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run baseline (heuristic)
```powershell
python baseline.py
```

3. Run baseline using OpenAI (optional)
```powershell
setx OPENAI_API_KEY "<your-key>"
python baseline.py
```

OpenEnv API

- `reset()` -> `env.models.Observation`
- `step(action: env.models.Action)` -> `(Observation, Reward, done: bool, info: dict)`
- `state()` -> serializable dict with internal state (e.g., phase)

Action / Observation / Reward
- Observation: `text`, `metadata`, `task_id`, `step`, `max_steps`.
- Action: `action_type` (e.g., `label`, `reply`, `summary`, `recommend`), `content`, `label`.
- Reward: `reward` float in [0.0, 1.0], with `debug` giving partial-credit diagnostics.

Tasks & Graders

- easy: single-label priority classification (partial credit for coarse match)
- medium: short reply addressing customer keywords + greeting (partial credit)
- hard: 3-phase incident handling (classify → summarize → remediation). Each phase gives partial reward; total normalized to [0,1].

Reward shaping

- Partial progress signals are provided by graders (overlap scores, length heuristics, phase-wise credit).
- Extra steps beyond `max_steps` are penalized.

Deployment & Docker

- The repository includes a `Dockerfile` (root). Build and run to verify the container. The `openenv.yaml` makes the environment discoverable for OpenEnv tooling and for Hugging Face Spaces deployment.

Notes & Next steps

- This is a first complete implementation focused on a useful domain (support/incident triage). Next improvements: larger example datasets, unit tests, `openenv validate` integration, a minimal web UI for HF Spaces, and more robust OpenAI prompt parsing.

If you'd like, I can:
- run `openenv validate` locally (if you want me to add a dev dependency)
- add unit tests for graders
- wire a minimal FastAPI endpoint for HF Space
