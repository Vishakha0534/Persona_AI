from unittest import result
from fastapi import FastAPI
from pydantic import BaseModel
from eval.environment import CareTriageEnv
from eval.models import Action
from fastapi.responses import HTMLResponse
from rapidfuzz import fuzz

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
                    margin-top: 60px;
                }
                h1 { font-size: 40px; }
                input {
                    padding: 10px;
                    margin: 6px;
                    border-radius: 6px;
                    border: none;
                    width: 250px;
                }
                button {
                    padding: 12px 25px;
                    background: #6366f1;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                }
                #result {
                    margin-top: 25px;
                    font-size: 18px;
                    line-height: 1.6;
                    background: #1e293b;
                    padding: 20px;
                    border-radius: 10px;
                    display: inline-block;
                    min-width: 300px;
                }
            </style>
        </head>
        <body>

            <h1>🤖 PersonaAI</h1>
            <p>Smart Healthcare Triage System</p>

            <input id="symptoms" placeholder="Enter symptoms" />
            <br>
            <input id="age" type="number" placeholder="Enter age" />
            <br><br>

            <button onclick="predict()">Analyze</button>

            <div id="result"></div>

            <script>
                async function predict() {
                    const symptoms = document.getElementById("symptoms").value;
                    const age = document.getElementById("age").value;

                    const res = await fetch('/triage', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            symptoms,
                            age: parseInt(age)
                        })
                    });

                    const data = await res.json();

                    document.getElementById("result").innerHTML = `
                        <b>Prediction:</b> ${data.prediction} <br>
                        <b>Confidence:</b> ${data.confidence} <br>
                        <b>Risk Score:</b> ${data.risk_score ?? "N/A"} <br>
                        <b>Hospital:</b> ${data.hospital ?? "Nearby Clinic"} <br>
                        <b>Advice:</b><br> ${data.advice}
                    `;
                }
            </script>

        </body>
    </html>
    """

# RESET ENVIRONMENT
@app.post("/reset")
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
    text = symptoms.lower().strip()

    # High-risk symptoms (weights)
    urgent_keywords = {
        "chest pain": 5,
        "breathing difficulty": 5,
        "unconscious": 6,
        "severe bleeding": 6,
        "heart attack": 7,
        "stroke": 7
    }

    # Moderate symptoms (weights)
    normal_keywords = {
        "fever": 2,
        "cough": 2,
        "cold": 1,
        "headache": 1,
        "vomiting": 2,
        "infection": 3
    }

    score = 0

    # fuzzy match function
    def is_match(keyword, text):
        return fuzz.partial_ratio(keyword, text) >= 80  # threshold

    # urgent matching (high impact)
    for word, weight in urgent_keywords.items():
        if is_match(word, text):
            score += weight

    # normal matching
    for word, weight in normal_keywords.items():
        if is_match(word, text):
            score += weight

    # age risk factor
    if age >= 60:
        score += 2
    elif age <= 10:
        score += 1

    # decision system
    if score >= 6:
        level = "urgent"
        confidence = min(0.90 + (score * 0.01), 0.99)
    elif score >= 2:
        level = "normal"
        confidence = 0.70 + (score * 0.02)
    else:
        level = "wait"
        confidence = 0.50 + (score * 0.02)

    return level, round(confidence, 2)

@app.post("/triage")
async def triage(data: dict):

    symptoms = data.get("symptoms", "")
    age = data.get("age", 0)

    result, confidence = triage_logic(symptoms, age)

    # 🔥 IMPROVED severity mapping (realistic scoring)
    severity_map = {
        "urgent": 3,
        "normal": 2,
        "wait": 1
    }

    severity_level = severity_map.get(result, 1)

    # 🔥 risk interpretation (frontend + evaluation useful)
    risk_score = (
        0.9 if result == "urgent"
        else 0.6 if result == "normal"
        else 0.3
    )

    # 🔥 structured advice engine (IMPORTANT UPGRADE)
    advice_map = {
        "urgent": """
🚨 URGENT CONDITION

• Visit nearest hospital immediately
• Avoid self-medication
• Seek emergency care
""",
        "normal": """
🩺 MODERATE CONDITION

• Consult doctor within 24–48 hours
• Rest and hydrate
• Monitor symptoms
""",
        "wait": """
💊 MILD CONDITION

• Rest at home
• Drink fluids
• Eat healthy food
• Monitor symptoms for 24–48 hours
"""
    }

    advice = advice_map[result]

    # 🔥 OPTIONAL: hospital hook (safe fallback)
    hospital = "Nearest hospital recommended"

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "severity_level": severity_level,
        "risk_score": risk_score,
        "hospital": hospital,
        "advice": advice.strip()
    }


# HELPER FUNCTION
def get_advice(level):
    if level == "urgent":
        return "Visit nearest hospital immediately."
    elif level == "normal":
        return "Consult a doctor soon."
    else:
        return "Monitor symptoms and rest."