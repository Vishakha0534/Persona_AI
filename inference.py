import sys
import os
import requests
from rapidfuzz import fuzz
from openai import OpenAI

# ---------------- SAFE ENV LOADING ----------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not API_BASE_URL or not API_KEY:
    print("[WARN] Missing API env vars - LLM will fallback", flush=True)

# ---------------- LLM CLIENT ----------------
client = None
if API_BASE_URL and API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )


# ---------------- INPUT CLEANING ----------------
def clean_input(text):
    text = text.strip().lower()
    text = " ".join(text.split())
    return text if len(text) > 1 else "fever"


# ---------------- RULE TRIAGE ENGINE ----------------
def rule_triage(text):

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

    score = 0
    matched = []

    def match(k):
        return k in text 

    # PRIORITY CHECK
    for k, w in urgent.items():
        if match(k):
            return "urgent", w, [k]   # immediate override

    for k, w in normal.items():
        if match(k):
            score += w
            matched.append(k)

    if score >= 3:
        level = "normal"
    else:
        level = "wait"

    return level, score, matched


# ---------------- TASK 2: RISK SCORE ----------------
def risk_score(score):
    return min(score / 10.0, 1.0)



# ---------------- NEAREST HOSPITAL (OPENSTREETMAP) ----------------
def get_nearest_hospital(lat, lon):
    try:
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:5000,{lat},{lon});
        out;
        """

        url = "https://overpass-api.de/api/interpreter"
        res = requests.get(url, params={"data": query}, timeout=5)
        data = res.json()

        hospitals = data.get("elements", [])

        if not hospitals:
            return "Nearby hospital not found"

        h = hospitals[0]
        return h.get("tags", {}).get("name", "Unknown Hospital")

    except:
        return "Hospital lookup unavailable"


# ---------------- RESPONSE ENGINE ----------------
def generate_action(level, symptoms, hospital_name):

    if level == "urgent":
        return f"""
🚨 URGENT CASE DETECTED

• Go immediately to: {hospital_name}
• Call emergency services if needed
• Avoid self-medication
• Keep patient stable
"""

    elif level == "normal":
        return f"""
🟡 NORMAL CASE

Home Care:
• Rest properly
• Drink fluids
• Steam inhalation
• Monitor 24–48 hours

If worsens → visit {hospital_name}
"""

    else:
        return f"""
🟢 MILD CASE

Health Tips:
• Healthy diet (fruits, vegetables)
• Light exercise
• Hydration
• Observe symptoms

If needed → visit {hospital_name}
"""


# ---------------- LLM REFINEMENT ----------------
def llm_refine(symptoms, rule_output):
    try:
        if not client:
            return rule_output, False

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY one word: urgent, normal, wait"},
                {"role": "user", "content": f"{symptoms} | rule={rule_output}"}
            ]
        )

        out = res.choices[0].message.content.strip().lower()

        if out not in ["urgent", "normal", "wait"]:
            return rule_output, False

        return out, True

    except:
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

        hospital = get_nearest_hospital(lat, lon) if lat and lon else "Nearby clinic"

        action = generate_action(final_level, symptoms, hospital)

        confidence = min(max(0.6 + score * 0.03, 0.5), 0.99)

        # ---------------- OPENENV OUTPUT ----------------
        print("[START] task=hospital_triage_system", flush=True)

        print(f"[STEP] symptoms={symptoms}", flush=True)
        print(f"[STEP] matched={matched}", flush=True)
        print(f"[STEP] rule_output={rule_level}", flush=True)
        print(f"[STEP] llm_output={llm_level} llm_ok={llm_ok}", flush=True)

        print(f"[STEP] prediction={final_level}", flush=True)
        print(f"[STEP] risk_score={round(risk,2)}", flush=True)
        print(f"[STEP] hospital={hospital}", flush=True)

        print(f"[STEP] recommendation={action}", flush=True)

        print(f"[END] task=hospital_triage_system result={final_level} score={round(confidence,2)}", flush=True)

    except Exception as e:
        print("[START] task=hospital_triage_system", flush=True)
        print(f"[STEP] error={str(e)}", flush=True)
        print("[END] task=hospital_triage_system result=wait score=0.5", flush=True)