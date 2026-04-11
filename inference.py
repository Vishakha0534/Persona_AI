import argparse

def analyze(symptoms, age):
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

    for word in urgent_keywords:
        if word in symptoms:
            score += 2

    for word in normal_keywords:
        if word in symptoms:
            score += 1

    # Age factor
    if age > 60:
        score += 1

    if score >= 3:
        return "urgent", 0.9
    elif score == 2:
        return "normal", 0.7
    else:
        return "wait", 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--symptoms", type=str, default="fever")
    parser.add_argument("--age", type=int, default=25)

    args = parser.parse_args()

    result, confidence = analyze(args.symptoms, args.age)

    print(f"{result} (confidence: {confidence})")