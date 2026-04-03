from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action(BaseModel):
    thought: str = Field(..., description="Internal reasoning before deciding on the action.")
    command: str = Field(..., description="Action command (e.g., search_kb, look_up_order, view_billing, issue_refund, grant_credit, reply)")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the command.")

class Observation(BaseModel):
    ticket_queue_size: int = Field(..., description="Number of unresolved tickets remaining.")
    current_ticket: Optional[Dict[str, Any]] = Field(None, description="Current ticket payload.")
    last_action_feedback: str = Field(..., description="System response from the previous action.")
    is_resolved: bool = Field(..., description="Whether the current ticket has been marked resolved.")

class Reward(BaseModel):
    value: float = Field(..., description="Reward value assigned.")
    reason: str = Field(..., description="A string detailing why the reward was given.")
