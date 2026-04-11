def extract_actions_and_messages(sample):
    actions = []
    msg = ""
    events = []

    try:
        # Aggressively extract the events list from ANY potential OpenEnv object type
        if isinstance(sample, dict):
            events = sample.get("action_history", sample.get("steps", sample.get("events", [])))
        elif isinstance(sample, list):
            events = sample
        elif hasattr(sample, "model_dump"):
            d = sample.model_dump()
            events = d.get("action_history", d.get("steps", d.get("events", [])))
        elif hasattr(sample, "steps"):
            events = sample.steps
        elif hasattr(sample, "action_history"):
            events = sample.action_history
    except Exception:
        pass

    for event in events:
        try:
            if isinstance(event, dict):
                act = event.get("action", {}) if "action" in event else event
                cmd = act.get("command", event.get("command"))
                if cmd:
                    actions.append(cmd)
                    if cmd == "reply":
                        args = act.get("args", event.get("args", {}))
                        if isinstance(args, dict):
                            msg += str(args.get("message", "")).lower() + " "
            elif hasattr(event, "action") or hasattr(event, "command"):
                act = getattr(event, "action", event)
                cmd = getattr(act, "command", getattr(event, "command", None))
                if cmd:
                    actions.append(cmd)
                    if cmd == "reply":
                        args = getattr(act, "args", getattr(event, "args", {}))
                        if isinstance(args, dict):
                            msg += str(args.get("message", "")).lower() + " "
        except Exception:
            continue

    return actions, msg

def _clamp(score):
    return max(0.01, min(0.99, float(score)))

def grade(sample, item=None):
    actions, msg = extract_actions_and_messages(sample)
    if "reply" in actions and "replace" in msg and ("sorry" in msg or "apolog" in msg):
        return _clamp(0.95)
    return _clamp(0.05)
