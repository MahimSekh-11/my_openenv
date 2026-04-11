def extract_actions_and_messages(sample):
    actions = []
    msg = ""
    # Support dict state natively without mocks
    if isinstance(sample, dict):
        action_hist = sample.get("action_history", [])
        for act in action_hist:
            cmd = act.get("command")
            if cmd:
                actions.append(cmd)
                if cmd == "reply":
                    args = act.get("args", {})
                    msg += str(args.get("message", "")).lower() + " "
                    
    # Support trajectory list
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

def grade_task_easy(sample=None, item=None):
    actions, msg = extract_actions_and_messages(sample)
    if "reply" in actions and ("replace" in msg) and ("apolog" in msg or "sorry" in msg):
        return 0.95
    return 0.05

def grade_task_medium(sample=None, item=None):
    actions, msg = extract_actions_and_messages(sample)
    if "reply" in actions and "30 days" in msg:
        if "look_up_order" in actions:
            return 0.95
        return 0.55
    return 0.05

def grade_task_hard(sample=None, item=None):
    actions, _ = extract_actions_and_messages(sample)
    if "issue_refund" in actions and "grant_credit" in actions:
        return 0.95
    return 0.05
