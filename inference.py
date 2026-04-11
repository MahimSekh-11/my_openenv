import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from support_ops.env import SupportOpsEnv

def clamp_fractional_score(value: float) -> float:
    """Keep emitted scores strictly inside (0, 1) for validator compatibility."""
    return max(0.01, min(0.99, float(value)))

def run_inference():
    # Automatically load environment variables from .env
    load_dotenv()
    
    # Load Environment Variables precisely as required
    api_key = os.getenv("HF_TOKEN", "dummy_token_for_validation") 
    if "your_actual_token_here" in api_key:
        api_key = "dummy_token_for_validation"
        
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
    # Initialize the required OpenAI client
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    tasks_to_run = ["task_easy", "task_medium", "task_hard"]
    
    for current_task in tasks_to_run:
        env = SupportOpsEnv()
        env.task_name = current_task
        
        # Output Mandatory START log
        print(f"[START] task={current_task}", flush=True)
        
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
                
                # Step the environment
                obs, reward, done, info = env.step(action)
                safe_reward = clamp_fractional_score(reward)
                rewards.append(safe_reward)
                
                # Output Mandatory STEP log format exactly
                print(f"[STEP] step={step_count} reward={safe_reward:.2f}", flush=True)
                
                # Attach to memory for the next loop
                messages.append({"role": "assistant", "content": raw_action})
                messages.append({"role": "user", "content": f"New Observation: {obs.model_dump_json()}"})
                
            except Exception as e:
                # Handle internal LLM parsing failure loop gracefully
                safe_reward = clamp_fractional_score(0.0)
                rewards.append(safe_reward)
                print(f"[STEP] step={step_count} reward={safe_reward:.2f}", flush=True)
                messages.append({"role": "user", "content": f"Format error: Please output valid JSON matching system instructions."})
        
        # Calculate final status
        final_score = clamp_fractional_score(env.score)
        
        # Output Mandatory END log exactly
        print(f"[END] task={current_task} score={final_score:.2f} steps={step_count}", flush=True)

if __name__ == "__main__":
    run_inference()
