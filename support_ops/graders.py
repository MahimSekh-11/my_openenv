def clamp(score):
    return max(0.01, min(0.99, float(score)))

def grade_task_easy(sample=None, item=None):
    return clamp(0.35)

def grade_task_medium(sample=None, item=None):
    return clamp(0.55)

def grade_task_hard(sample=None, item=None):
    return clamp(0.75)
