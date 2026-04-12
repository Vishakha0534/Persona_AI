def easy(state, action, result):
    """
    Easy task:
    Focus: basic correctness + survival of episode
    """
    if result.get("done"):
        return 0.7
    if result.get("error"):
        return 0.2
    return 0.5


def medium(state, action, result):
    """
    Medium task:
    Focus: progress across steps + correct decisions
    """
    step = state.get("step_count", 0)

    base = 0.3

    if step > 1:
        base += 0.2
    if step > 3:
        base += 0.2

    if result.get("done"):
        base += 0.2

    if result.get("error"):
        base -= 0.2

    return round(max(0.0, min(1.0, base)), 2)


def hard(state, action, result):
    """
    Hard task:
    Focus: efficiency + completion under constraints
    """
    step = state.get("step_count", 0)

    reward = 0.4

    # reward efficiency (fewer steps better)
    if step <= 3:
        reward += 0.3
    elif step <= 5:
        reward += 0.2
    else:
        reward += 0.1

    # completion bonus
    if result.get("done"):
        reward += 0.3

    # penalty for invalid actions
    if result.get("error"):
        reward -= 0.2

    return round(max(0.0, min(1.0, reward)), 2)