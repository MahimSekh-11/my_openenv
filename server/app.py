from fastapi import FastAPI
from support_ops.env import SupportOpsEnv

# Required for HF Space deployment validation
# Pings to the space will hit `/` returning 200, and /reset correctly initializes.

app = FastAPI(title="SupportOps OpenEnv API")
env = SupportOpsEnv()

@app.get("/")
def health_check():
    return {"status": "SupportOps Space is running"}

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs.model_dump()

@app.get("/state")
def get_state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

