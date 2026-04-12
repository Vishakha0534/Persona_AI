import random
import sys
import grader
from models import Patient, Action


class CareTriageEnv:
    def __init__(self):
        self.max_steps = 6
        self.task = None
        self.step_count = 0

        # ✅ REQUIRED: 3 TASKS WITH GRADERS
        self.tasks = {
            "easy": grader.easy,
            "medium": grader.medium,
            "hard": grader.hard
        }

    # -------------------------
    # RESET
    # -------------------------
    async def reset(self, task="easy"):
        self.task = task
        self.step_count = 0
        self.available_beds = 2

        if task == "easy":
            self.patients = [
                Patient(id=1, symptoms="chest pain", severity=9, assigned=False),
                Patient(id=2, symptoms="fever", severity=3, assigned=False),
            ]

        elif task == "medium":
            self.patients = [
                Patient(id=1, symptoms="breathing issue", severity=8, assigned=False),
                Patient(id=2, symptoms="high fever", severity=6, assigned=False),
                Patient(id=3, symptoms="cold", severity=2, assigned=False),
            ]

        else:  # hard
            self.patients = [
                Patient(id=1, symptoms="heart attack", severity=9, assigned=False),
                Patient(id=2, symptoms="lung issue", severity=7, assigned=False),
                Patient(id=3, symptoms="fever", severity=5, assigned=False),
                Patient(id=4, symptoms="headache", severity=2, assigned=False),
            ]

        return self._response(0.0, False, None)

    # -------------------------
    # STEP
    # -------------------------
    async def step(self, action: Action):
        self.step_count += 1

        reward = 0.0
        error = None

        # find patient
        patient = next((p for p in self.patients if p.id == action.patient_id), None)

        if not patient:
            error = "invalid patient"
            reward = 0.0

        elif patient.assigned:
            error = "already assigned"
            reward = 0.1

        else:
            # partial rewards
            if "pain" in patient.symptoms:
                reward += 0.3
            if "fever" in patient.symptoms:
                reward += 0.2
            if "breathing" in patient.symptoms:
                reward += 0.3

            # assign bed logic
            if action.assign_bed and self.available_beds > 0:
                patient.assigned = True
                self.available_beds -= 1

                reward += 0.5 if patient.severity > 7 else 0.2
            else:
                reward += 0.1

        # normalize reward
        reward = round(min(reward, 1.0), 2)

        # dynamic hard mode behavior
        if self.task == "hard" and random.random() < 0.3:
            new_id = len(self.patients) + 1
            self.patients.append(
                Patient(
                    id=new_id,
                    symptoms="emergency",
                    severity=random.randint(3, 9),
                    assigned=False,
                )
            )

        # done condition
        done = self.step_count >= self.max_steps or self.available_beds == 0

        return self._response(reward, done, error)

    # -------------------------
    # STATE
    # -------------------------
    async def state(self):
        return self._get_obs()

    async def close(self):
        pass

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _get_obs(self):
        return {
            "patients": [
                {
                    "id": p.id,
                    "symptoms": p.symptoms,
                    "severity": p.severity,
                    "assigned": p.assigned
                }
                for p in self.patients
            ],
            "available_beds": self.available_beds,
            "step_count": self.step_count,
            "task": self.task,
        }

    # -------------------------
    # RESPONSE WRAPPER
    # -------------------------
    def _response(self, reward, done, error=None):
        return {
            "observation": self._get_obs(),
            "reward": float(reward),
            "done": bool(done),
            "info": {
                "error": error
            }
        }