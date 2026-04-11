from unittest import result
from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import CareTriageEnv
from env.models import Action
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="PersonaAI - Healthcare Triage API",
    description="AI-powered triage system for symptom analysis",
    version="1.0"
)

env = CareTriageEnv()

# HEALTH CHECK
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>PersonaAI</title>
            <style>
                body {
                    font-family: Arial;
                    text-align: center;
                    background: #0f172a;
                    color: white;
                    margin-top: 80px;
                }
                h1 {
                    font-size: 40px;
                }
                input {
                    padding: 10px;
                    margin: 5px;
                    border-radius: 5px;
                    border: none;
                }
                button {
                    padding: 10px 20px;
                    background: #6366f1;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                }
                #result {
                    margin-top: 20px;
                    font-size: 18px;
                }
            </style>
        </head>
        <body>
            <h1>🤖 PersonaAI</h1>
            <p>Healthcare AI Triage System</p>

            <input id="symptoms" placeholder="Enter symptoms" />
            <br>
            <input id="age" type="number" placeholder="Enter age" />
            <br><br>

            <button onclick="predict()">Check Triage</button>

            <div id="result"></div>

            <br><br>
            <a href="/docs" style="color: lightblue;">Open API Docs</a>

            <script>
                async function predict() {
                    const symptoms = document.getElementById("symptoms").value;
                    const age = document.getElementById("age").value;

                    const res = await fetch('/triage', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symptoms, age: parseInt(age) })
                    });

                    const data = await res.json();

                    document.getElementById("result").innerHTML =
                        "Prediction: " + data.prediction +
                        "<br>Confidence: " + data.confidence +
                        "<br>Advice: " + data.advice;
                }
            </script>
        </body>
    </html>
    """

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

    # 🔴 High-risk symptoms (higher weights)
    urgent_keywords = {
        "chest pain": 5,
        "breathing difficulty": 5,
        "unconscious": 6,
        "severe bleeding": 6,
        "heart attack": 7,
        "stroke": 7
    }

    # 🟡 Moderate symptoms
    normal_keywords = {
        "fever": 2,
        "cough": 2,
        "cold": 1,
        "headache": 1,
        "vomiting": 2,
        "infection": 3
    }

    score = 0

    # 🔴 Check urgent
    for word, weight in urgent_keywords.items():
        if word in symptoms:
            score += weight

    # 🟡 Check normal
    for word, weight in normal_keywords.items():
        if word in symptoms:
            score += weight

    # 👴 Age factor
    if age >= 60:
        score += 2
    elif age <= 10:
        score += 1

    # 🧠 Decision thresholds
    if score >= 6:
        level = "urgent"
        confidence = min(0.9 + score * 0.01, 0.99)
    elif score >= 2:
        level = "normal"
        confidence = 0.7 + score * 0.02
    else:
        level = "wait"
        confidence = 0.5 + score * 0.02

    return level, round(confidence, 2)

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