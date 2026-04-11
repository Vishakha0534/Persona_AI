import sys
import os
from rapidfuzz import fuzz
from openai import OpenAI

# ---------------- LLM CLIENT (OPENENV PROXY) ----------------
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

# ---------------- INPUT SANITIZATION (FIX 1) ----------------
def clean_input(text):
    text = text.strip().lower()
    text = " ".join(text.split())

    # remove junk inputs
    if len(text) < 2:
        return "fever"

    return text


# ---------------- RULE ENGINE ----------------
def rule_triage(text):
    urgent_signals = {
        "chest pain": 3,
        "breathing difficulty": 3,
        "unconscious": 4,
        "severe bleeding": 4,
        "heart attack": 5,
        "stroke": 5
    }

    normal_signals = {
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
        return fuzz.partial_ratio(k, text) >= 80

    for k, w in urgent_signals.items():
        if match(k):
            score += w
            matched.append(k)

    for k, w in normal_signals.items():
        if match(k):
            score += w
            matched.append(k)

    if score >= 7:
        level = "urgent"
    elif score >= 3:
        level = "normal"
    else:
        level = "wait"

    return level, score, matched


# ---------------- STRICT LLM CALL (FIX 2: CONTROL OUTPUT) ----------------
def llm_refine(symptoms, rule_output):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY one word: urgent OR normal OR wait. "
                        "No explanation. No punctuation."
                    )
                },
                {
                    "role": "user",
                    "content": f"{symptoms} | rule={rule_output}"
                }
            ]
        )

        out = response.choices[0].message.content.strip().lower()

        # safety filter (VERY IMPORTANT FIX)
        if out not in ["urgent", "normal", "wait"]:
            return rule_output, False

        return out, True

    except:
        return rule_output, False


# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        raw_input = " ".join(sys.argv[1:])
        symptoms = clean_input(raw_input)

        # STEP 1: rule system
        rule_level, score, matched = rule_triage(symptoms)

        # STEP 2: LLM refinement
        llm_level, llm_ok = llm_refine(symptoms, rule_level)

        # STEP 3: final decision (consistency logic FIX 3)
        if llm_ok and llm_level == rule_level:
            final_level = llm_level
        elif llm_ok:
            final_level = llm_level  # trust LLM if valid
        else:
            final_level = rule_level  # fallback safe

        # STEP 4: confidence (FIX 4 - more realistic)
        base_conf = 0.6 + (score * 0.03)
        llm_boost = 0.1 if llm_ok else -0.05
        confidence = min(max(base_conf + llm_boost, 0.5), 0.99)

        # ---------------- OPENENV FORMAT ----------------
        print("[START] task=triage", flush=True)
        print(f"[STEP] input={symptoms}", flush=True)
        print(f"[STEP] matched={matched}", flush=True)
        print(f"[STEP] rule_output={rule_level}", flush=True)
        print(f"[STEP] llm_output={llm_level} llm_ok={llm_ok}", flush=True)
        print(f"[STEP] final_prediction={final_level}", flush=True)
        print(f"[END] task=triage result={final_level} score={round(confidence,2)}", flush=True)

    except Exception as e:
        print("[START] task=triage", flush=True)
        print(f"[STEP] error={str(e)}", flush=True)
        print("[END] task=triage result=wait score=0.5", flush=True)