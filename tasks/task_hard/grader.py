def extract_actions_and_messages(sample):
    actions = []
    msg = ""
    # Support dict state
    if isinstance(sample, dict):
        action_hist = sample.get("action_history", [])
        for act in action_hist:
            cmd = act.get("command")
            if cmd:
                actions.append(cmd)
                if cmd == "reply":
                    args = act.get("args", {})
                    msg += str(args.get("message", "")).lower() + " "
                    
    # Support trajectory list commonly used by OpenEnv validators
    elif isinstance(sample, list):
        for event in sample:
            if isinstance(event, dict):
                act = event.get("action", {}) if event.get("type") == "action" else event
                cmd = act.get("command", event.get("command"))
                if cmd:
                    actions.append(cmd)
                    if cmd == "reply":
                        args = act.get("args", event.get("args", {}))
                        msg += str(args.get("message", "")).lower() + " "
    return actions, msg

def _clamp(score):
    return max(0.01, min(0.99, float(score)))

def grade(sample, item=None):
    actions, msg = extract_actions_and_messages(sample)
    if "issue_refund" in actions and "grant_credit" in actions:
        return _clamp(0.95)
    return _clamp(0.05)
