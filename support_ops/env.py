import os
from typing import Tuple, Dict, Any, Optional, List
from support_ops.schemas import Action, Observation, Reward

class SupportOpsEnv:
    task_name: str
    kb: Dict[str, str]
    orders: Dict[str, Dict[str, Any]]
    billing: Dict[str, List[Dict[str, Any]]]
    tickets: Dict[str, Dict[str, Any]]
    current_ticket: Optional[Dict[str, Any]]
    resolved: bool
    score: float
    actions_taken: List[str]

    def __init__(self):
        # Allow specific task loading, defaulting to easy
        self.task_name = os.getenv("MY_ENV_V4_TASK", "task_easy")
        
        # Mock internal Knowledge Base
        self.kb = {
            "missing_item": "If a user reports a missing item within 7 days, issue a replacement and apologize in reply.",
            "refund_policy": "Returns allowed strictly within 30 days of delivery. Reject any return after 30 days.",
            "double_charge": "April Glitch SOP: If user reports double charges, verify duplicate $15.99 charges in billing. Refund one charge, grant 1 month system credit, and apologize."
        }
        
        # Mock Order DB
        self.orders = {
            "O-100": {"delivery_days_ago": 2, "items": ["Mic", "Missing_Cable"]},
            "O-200": {"delivery_days_ago": 45, "items": ["Laptop"]}
        }
        
        # Mock Billing Ledger
        self.billing = {
            "U-300": [{"id": "C-1", "amount": 15.99}, {"id": "C-2", "amount": 15.99}]
        }
        
        # Tasks queue
        self.tickets = {
            "task_easy": {"id": "T-001", "user_id": "U-100", "body": "Order O-100 has a missing USB cable. Need a replacement.", "issue": "missing"},
            "task_medium": {"id": "T-002", "user_id": "U-200", "body": "Want to return Order O-200. I don't need it.", "issue": "return"},
            "task_hard": {"id": "T-003", "user_id": "U-300", "body": "I looked at my bill and there are two $15.99 charges!", "issue": "billing"}
        }

        self.reset()

    def reset(self) -> Observation:
        """Resets to initial observation per OpenEnv Spec"""
        self.current_ticket = self.tickets.get(self.task_name, self.tickets["task_easy"])
        self.resolved = False
        self.score = 0.0
        self.actions_taken = []
        return self._get_obs("Environment starting. You have 1 unread ticket in queue.")

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Process agent action, strictly shaping dense rewards per rules"""
        cmd = action.command
        args = action.args
        feedback = ""
        reward_delta = 0.0
        done = False
        
        self.actions_taken.append(cmd)

        try:
            if cmd == "read_ticket":
                feedback = f"Ticket Details: {self.current_ticket}"
                if "read_ticket" not in self.actions_taken[:-1]:
                    reward_delta += 0.1
            
            elif cmd == "search_kb":
                query = str(args.get("query", "")).lower().replace("_", " ")
                found = False
                for k, v in self.kb.items():
                    k_clean = k.replace("_", " ")
                    # Match if key is in query, or query keywords match content
                    if k_clean in query or any(word in query for word in k_clean.split()):
                        feedback = f"KB Article Found [{k}]: {v}"
                        found = True
                        if "search_kb" not in self.actions_taken[:-1]:
                            reward_delta += 0.1
                        break
                if not found:
                    feedback = "System Search: No KB articles matched your query. Try different keywords."
            
            elif cmd == "look_up_order":
                oid = args.get("order_id")
                if oid in self.orders:
                    feedback = f"DB Order Record {oid}: {self.orders[oid]}"
                    if "look_up_order" not in self.actions_taken[:-1]:
                        reward_delta += 0.1
                else:
                    feedback = f"DB Error: Order {oid} not found."
            
            elif cmd == "view_billing":
                uid = args.get("user_id")
                if uid in self.billing:
                    feedback = f"DB Billing Ledger {uid}: {self.billing[uid]}"
                    if "view_billing" not in self.actions_taken[:-1]:
                        reward_delta += 0.1
                else:
                    feedback = f"DB Error: User {uid} not found."
                    
            elif cmd == "issue_refund":
                if self.task_name == "task_hard" and str(args.get("amount")) == "15.99":
                    feedback = "System: $15.99 Refund issued successfully."
                    if "issue_refund" not in self.actions_taken[:-1]:
                        reward_delta += 0.2
                else:
                    feedback = "System Warning: Refund parameters invalid or not permitted."
                    reward_delta -= 0.1
                    
            elif cmd == "grant_credit":
                if self.task_name == "task_hard" and str(args.get("months")) == "1":
                    feedback = "System: 1 Month credit granted successfully."
                    if "grant_credit" not in self.actions_taken[:-1]:
                        reward_delta += 0.2
                else:
                    feedback = "System Warning: Credit failed or invalid."
                    reward_delta -= 0.1

            elif cmd == "reply":
                msg = args.get("message", "").lower()
                done = True
                self.resolved = True
                feedback = f"System: Message sent to customer. Ticket closed."
                
                # Top up to exactly 1.0 on correct resolution
                target_balance = max(0.0, 1.0 - self.score)
                if self.task_name == "task_easy":
                    if "replace" in msg and ("apolog" in msg or "sorry" in msg):
                        reward_delta += target_balance
                        
                elif self.task_name == "task_medium":
                    if "30 days" in msg:
                        if "look_up_order" in self.actions_taken:
                            reward_delta += target_balance
                        else:
                            reward_delta += 0.3 # Guessed right but missed procedure
                            
                elif self.task_name == "task_hard":
                    if "issue_refund" in self.actions_taken and "grant_credit" in self.actions_taken:
                         reward_delta += target_balance

            else:
                feedback = "Agent System Error: Unknown command."
                reward_delta -= 0.1

        except Exception as e:
            feedback = f"Exception executing action parameters: {str(e)}"
            reward_delta -= 0.5 

        self.score += reward_delta
        # Clamp environment output values
        self.score = max(0.0, min(1.0, self.score))
        return self._get_obs(feedback), reward_delta, done, {"reason": feedback}

    def _get_obs(self, feedback: str) -> Observation:
        return Observation(
            ticket_queue_size=1 if not self.resolved else 0,
            current_ticket=self.current_ticket if not self.resolved else None,
            last_action_feedback=feedback,
            is_resolved=self.resolved
        )

    def state(self):
        """Current state dumping for full observability"""
        return {
            "score": self.score,
            "actions_taken": self.actions_taken,
            "resolved": self.resolved,
            "task": self.task_name
        }
