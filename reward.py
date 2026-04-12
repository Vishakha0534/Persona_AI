def compute_reward(patient):
    base = patient.severity / 10.0
    if patient.severity >= 8:
        return min(1.0, base + 0.3)
    elif patient.severity >= 5:
        return base
    else:
        return base - 0.4