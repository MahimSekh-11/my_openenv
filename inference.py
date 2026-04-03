import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from support_ops.env import SupportOpsEnv

def run_inference():
    # Automatically load environment variables from .env
    load_dotenv()
    
    # Load Environment Variables precisely as required
    api_key = os.getenv("HF_TOKEN") 
    if not api_key or "your_actual_token_here" in api_key:
        print("Error: API Key (HF_TOKEN) is missing. Please add it to your .env file.")
        return
        
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    task_name = os.getenv("MY_ENV_V4_TASK", "task_easy")
    benchmark = os.getenv("MY_ENV_V4_BENCHMARK", "support_ops")
    
    # Initialize the required OpenAI client
    client = OpenAI(api_key=api_key, base_url=api_base)
    env = SupportOpsEnv()
    
    # Output Mandatory START log
    print(f"[START] task={task_name} env={benchmark} model={model_name}")
    
    obs = env.reset()
    done = False
    step_count = 0
    rewards = []
    
    # Strict prompt schema guiding the LLM
    system_prompt = """You are an automated Customer Support agent. 
You must follow company policy strictly. ALWAYS check the Knowledge Base (`search_kb`) for the relevant SOP before taking actions or replying. 

You must output a JSON object with a 'thought' field for reasoning and a 'command' field for the action. 
Example formats:
{"thought": "I need to check the missing item policy.", "command": "search_kb", "args": {"query": "missing item"}} 
{"thought": "Now I verify the order details.", "command": "look_up_order", "args": {"order_id": "O-100"}}
{"thought": "The policy says to apologize and issue a replacement.", "command": "reply", "args": {"message": "I apologize..."}}

Step-by-step guidance:
1. Search the KB to find the policy for the specific issue.
2. Verify order/billing details in the database if the policy requires it.
3. Perform any required system actions (refund/credit/replacement).
4. Reply to the customer following the exact language required by the policy."""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"New Observation: {obs.model_dump_json()}"}]
                
    while not done and step_count < 10: # Circuit breaker to prevent 20min timeout violation
        step_count += 1
        raw_action = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            raw_action = response.choices[0].message.content
            action_data = json.loads(raw_action)
            
            # Map LLM JSON output to typed Pydantic OpenEnv Action Schema
            from support_ops.schemas import Action
            action = Action(**action_data)
            action_str = f"thought='{action.thought}', {action.command}({json.dumps(action.args)})"
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            # Output Mandatory STEP log format exactly
            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # Attach to memory for the next loop
            messages.append({"role": "assistant", "content": raw_action})
            messages.append({"role": "user", "content": f"New Observation: {obs.model_dump_json()}"})
            
        except Exception as e:
            # Handle internal LLM parsing failure loop gracefully
            err_msg = str(e).replace('"', "'").replace("\n", "")
            print(f"[STEP] step={step_count} action=parse_fail reward=0.00 done=false error=\"{err_msg}\"", flush=True)
            messages.append({"role": "user", "content": f"Format error: Please output valid JSON matching system instructions."})
    
    # Calculate final status
    success = str(env.score >= 0.9).lower()
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    
    # Output Mandatory END log exactly
    print(f"[END] success={success} steps={step_count} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_inference()
