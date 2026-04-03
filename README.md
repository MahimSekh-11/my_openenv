---
title: SupportOps OpenEnv
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# SupportOps: OpenEnv CS Triage & Resolution Simulator 

## Motivation & Domain
Customer Support (Tier 1 & Tier 2) represents a multi-billion dollar automation target. An LLM agent operating effectively as a support engineer needs the ability to:
1. Absorb incoming request context correctly.
2. Search standard operating procedures (SOPs) based on keyword intuition.
3. Verify claims against real databases (Order delivery logs, Billing ledgers). 
4. Execute mutations (Issue Refunds, Grant Credits) intelligently and deterministically contextually before replying.

This OpenEnv correctly simulates this reality. It provides a stringent grading system minimizing "prompt hacking" by requiring exact action sequences to verify SOPs internally, providing partial (dense) rewards for successful context tracking along the way.

## Action Space
Agent issues commands via Pydantic matching dictionary formats:
- `read_ticket` (Extract initially loaded ticket payload)
- `search_kb` (Query mock document indexing to understand company rule logic)
- `look_up_order` (Verify order delivery dates/timelines)
- `view_billing` (Verify ledger transactions)
- `issue_refund` (Modify db state ledger)
- `grant_credit` (Modify db state ledger)
- `reply` (Triggers closing condition with final string)

## Observation Space
- `ticket_queue_size`: (int) Tracking queue draining progress.
- `current_ticket`: (dict) Exposing the ID, User string, and text bodies.
- `last_action_feedback`: (str) Real-time query result simulation.
- `is_resolved`: (bool) Close terminal state check.

## Setup & Validation
This package provides a standalone HF space deployable environment that adheres strictly to `openenv-core` expectations.

```bash
# Docker testing
docker build -t support_ops_env .
docker run -p 7860:7860 support_ops_env

# Baseline Execution Test 
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-token>"
MY_ENV_V4_TASK="task_easy" py inference.py
MY_ENV_V4_TASK="task_medium" py inference.py
MY_ENV_V4_TASK="task_hard" py inference.py
```
