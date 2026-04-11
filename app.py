from unittest import result

from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import CareTriageEnv
from env.models import Action

app = FastAPI(
    title="PersonaAI - Healthcare Triage API",
    description="AI-powered triage system for symptom analysis",
    version="1.0"
)

env = CareTriageEnv()

# HEALTH CHECK
@app.get("/")
def home():
    return {
        "status": "running",
        "project": "PersonaAI",
        "message": "Healthcare AI Triage System Active"
    }


# RESET ENVIRONMENT
@app.get("/reset")
async def reset(task: str = "easy"):
    return await env.reset(task)

# STEP (AGENT ACTION)
@app.post("/step")
async def step(action: dict):
    act = Action(**action)
    return await env.step(act)

# GET CURRENT STATE
@app.get("/state")
async def state():
    return await env.state()

# DIRECT TRIAGE API 
class TriageInput(BaseModel):
    symptoms: str
    age: int

def triage_logic(symptoms, age):
    symptoms = symptoms.lower()

    urgent_keywords = [
        "chest pain", "breathing difficulty", "unconscious",
        "severe bleeding", "heart attack", "stroke"
    ]

    normal_keywords = [
        "fever", "cough", "cold", "headache",
        "vomiting", "infection"
    ]

    score = 0

    # urgent detection
    for word in urgent_keywords:
        if word in symptoms:
            return "urgent", 0.95   # immediate return

    # Normal scoring
    for word in normal_keywords:
        if word in symptoms:
            score += 1

    # Age factor
    if age > 60:
        score += 1

    if score >= 2:
        return "normal", 0.7
    else:
        return "wait", 0.5


@app.post("/triage")
async def triage(data: dict):
    symptoms = data.get("symptoms", "")
    age = data.get("age", 0)

    result, confidence = triage_logic(symptoms, age)

    return {
    "prediction": result,
    "confidence": confidence,
    "severity_level": 3 if result == "urgent" else 2 if result == "normal" else 1,
    "advice": (
        "🚨 Go to nearest hospital immediately"
        if result == "urgent"
        else "🩺 Consult a doctor soon"
        if result == "normal"
        else "💊 Rest and monitor symptoms"
    )
}


# HELPER FUNCTION
def get_advice(level):
    if level == "urgent":
        return "Visit nearest hospital immediately."
    elif level == "normal":
        return "Consult a doctor soon."
    else:
        return "Monitor symptoms and rest."