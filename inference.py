import sys
import os
import requests
from rapidfuzz import fuzz
from openai import OpenAI

def clean_text(text):
    return " ".join(text.lower().strip().split())


# ---------------- SAFE ENV ----------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = None
if API_BASE_URL and API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
else:
    print("[WARN] LLM disabled - using rule system only", flush=True)


# ---------------- CLEAN INPUT ----------------
def clean_input(text):
    text = text.strip().lower()
    text = " ".join(text.split())
    return text if len(text) > 1 else "fever"


# ---------------- RULE ENGINE (FIXED & SAFE) ----------------
def rule_triage(text):
    text = clean_text(text)

    urgent = {
        "chest pain": 5,
        "breathing difficulty": 5,
        "unconscious": 6,
        "severe bleeding": 6,
        "heart attack": 7,
        "stroke": 7
    }

    normal = {
        "fever": 2,
        "cough": 2,
        "cold": 1,
        "headache": 1,
        "vomiting": 2,
        "infection": 2
    }

    text = text.lower()

    matched_urgent = []
    matched_normal = []
    score = 0

    # 🔥 STEP 1: STRICT URGENT PRIORITY CHECK (FIX)
    for k, w in urgent.items():
        if k in text:
            matched_urgent.append(k)
            score += w

    # 🚨 IF ANY URGENT FOUND → DIRECT RETURN
    if matched_urgent:
        return "urgent", score, matched_urgent

    # 🟡 STEP 2: NORMAL ONLY IF NO URGENT
    for k, w in normal.items():
        if k in text:
            matched_normal.append(k)
            score += w

    if score >= 3:
        return "normal", score, matched_normal

    return "wait", score, matched_normal

# ---------------- RISK SCORE (STABLE) ----------------
def risk_score(score):
    return min(score / 10.0, 1.0)


# ---------------- HOSPITAL LOOKUP (SAFE FALLBACK) ----------------
def get_nearest_hospital(lat, lon):
    try:
        if lat is None or lon is None:
            return "Nearby clinic"

        query = f"""
        [out:json];
        node["amenity"="hospital"](around:5000,{lat},{lon});
        out;
        """

        url = "https://overpass-api.de/api/interpreter"
        res = requests.get(url, params={"data": query}, timeout=5)
        data = res.json()

        hospitals = data.get("elements", [])

        if hospitals:
            return hospitals[0].get("tags", {}).get("name", "Unknown Hospital")

    except:
        pass

    return "District Hospital (fallback)"


# ---------------- RESPONSE ENGINE ----------------
def generate_action(level, hospital):

    if level == "urgent":
        return f"""
🚨 URGENT CASE

• Go immediately to: {hospital}
• Call emergency services
• Do NOT ignore symptoms
"""

    elif level == "normal":
        return f"""
🟡 NORMAL CASE

• Rest well
• Hydrate properly
• Monitor symptoms

If worse → visit {hospital}
"""

    else:
        return f"""
🟢 MILD CASE

• Healthy diet
• Light exercise
• Observe symptoms

If needed → visit {hospital}
"""


# ---------------- LLM REFINE (SAFE) ----------------
def llm_refine(symptoms, rule_output):

    if not client:
        return rule_output, False

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY one word: urgent, normal, wait"},
                {"role": "user", "content": symptoms}
            ]
        )

        out = res.choices[0].message.content.strip().lower()

        if out in ["urgent", "normal", "wait"]:
            return out, True

    except:
        pass

    return rule_output, False


# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        args = sys.argv[1:]

        lat, lon = None, None
        words = []

        i = 0
        while i < len(args):
            if args[i] == "--lat":
                lat = float(args[i+1])
                i += 2
            elif args[i] == "--lon":
                lon = float(args[i+1])
                i += 2
            else:
                words.append(args[i])
                i += 1

        symptoms = clean_input(" ".join(words))

        # ---------------- PIPELINE ----------------
        rule_level, score, matched = rule_triage(symptoms)

        risk = risk_score(score)

        llm_level, llm_ok = llm_refine(symptoms, rule_level)

        final_level = llm_level if llm_ok else rule_level

        hospital = get_nearest_hospital(lat, lon)

        action = generate_action(final_level, hospital)

        confidence = {
            "urgent": 0.95,
            "normal": 0.75,
            "wait": 0.55
        }.get(final_level, 0.6)

        # ---------------- OPENENV OUTPUT ----------------
        print("[START] task=hospital_triage_system", flush=True)

        print(f"[STEP] task_1=triage_classification", flush=True)
        print(f"[STEP] task_2=risk_scoring", flush=True)
        print(f"[STEP] task_3=hospital_recommendation", flush=True)

        print(f"[STEP] symptoms={symptoms}", flush=True)
        print(f"[STEP] matched={matched}", flush=True)
        print(f"[STEP] rule_output={rule_level}", flush=True)
        print(f"[STEP] llm_output={llm_level} llm_ok={llm_ok}", flush=True)

        print(f"[STEP] prediction={final_level}", flush=True)
        print(f"[STEP] risk_score={round(risk,2)}", flush=True)
        print(f"[STEP] hospital={hospital}", flush=True)

        print(f"[STEP] recommendation={action}", flush=True)

        print(f"[END] task=hospital_triage_system result={final_level} score={confidence}", flush=True)

    except Exception as e:
        print("[START] task=hospital_triage_system", flush=True)
        print(f"[STEP] error={str(e)}", flush=True)
        print("[END] task=hospital_triage_system result=wait score=0.5", flush=True)