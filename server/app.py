from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from support_ops.env import SupportOpsEnv
from support_ops.schemas import Action, Observation

app = FastAPI(title="SupportOps OpenEnv API")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

env = SupportOpsEnv()

@app.get("/", response_class=HTMLResponse)
def root_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return """
    <html>
      <head><title>SupportOps OpenEnv API</title></head>
      <body>
        <h1>SupportOps OpenEnv API</h1>
        <p>The environment server is running.</p>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "support_ops",
        "description": "Tier 1 Customer Support Triaging and Management Environment",
    }

@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "actions_taken": {"type": "array", "items": {"type": "string"}},
                "resolved": {"type": "boolean"},
                "task": {"type": "string"},
            },
            "required": ["score", "actions_taken", "resolved", "task"],
        },
    }

@app.post("/reset")
def reset_env_top():
    # Maintain openenv validator compatibility
    return env.reset().model_dump()

@app.get("/state")
def get_state_top():
    return env.state()

# GUI API Routes
@app.post("/api/reset")
def reset_env():
    obs = env.reset()
    return {"observation": obs.model_dump(), "state": env.state()}

@app.post("/api/step")
def step_env(action: Action):
    obs, reward_delta, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward_delta,
        "done": done,
        "info": info,
        "state": env.state()
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
