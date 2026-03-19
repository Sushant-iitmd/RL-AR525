# AR525 Assignment 2 — Drone Hovering with Model-Free RL (Report)

**Course:** AR-525: Reinforcement Learning for Robotics
**Instructor:** Dr. Deepak Raina, CAIR, IIT Mandi
**Submitted by:** Sushant (S25064) · Harsh Vardhan Saxena (S25068)
**Date:** March 19, 2026

---

## 1. Objectives

The assignment required implementing two model-free RL algorithms from scratch and using them to train a drone to hover at a fixed target position `[0, 0, 1]` in the `HoverAviary` PyBullet simulator.

### Required Tasks

| Task | Description | Weight |
|------|-------------|--------|
| Monte Carlo Control | First-visit MC with ε-greedy exploration, backward discounted return, Q-table update | 30% |
| Q-Learning | Off-policy TD with `max Q(s',a')` update, ε-greedy selection | 30% |
| Experiments | Hyperparameter tuning, learning curves, algorithm comparison | 25% |
| Code Quality | Readability, naming, comments | 15% |

### Bonus Challenges

| Challenge | Points | Requirement |
|-----------|--------|-------------|
| SARSA | 5 pts | On-policy TD using the actual next action |
| Double Q-Learning | 7 pts | Two Q-tables to reduce maximization bias |
| Experience Replay | 8 pts | Replay buffer with random mini-batch updates |

Bonus points are awarded when evaluation score ≥ 300. Completing all three gives **+20 bonus points**.

---

## 2. What We Achieved

### Core Tasks — Complete

**Monte Carlo Control** is implemented with first-visit semantics. The episode trajectory is collected in full, then walked backwards to accumulate the discounted return `G = r + γ·G`. Only the first occurrence of each `(state, action)` pair in a given episode contributes to the update, which avoids double-counting repeated visits. The incremental `α`-weighted update rule is used instead of a strict sample average to handle the non-stationarity of a changing policy.

**Q-Learning** is implemented as off-policy TD control. At every step, the Q-value is updated using the Bellman optimality target — the reward plus the discounted maximum Q-value over all actions in the next state. The behaviour policy is ε-greedy, but the update always uses `max`, making the algorithm off-policy and faster-converging than on-policy methods on this task.

Both implementations correctly use `format_action()` to pass the action as a `(1,1)` float array to `env.step()` and `extract_position()` to pull the relevant `(x, y, z)` slice from the raw kinematic observation.

### Bonus Challenges — All 3 Complete (+20 pts)

**SARSA** — On-policy TD using the actual next epsilon-greedy action. Uses epsilon decay from 0.5 → 0.01 with best-Q-table checkpointing every 50 episodes. Final eval: **454.46**.

**Double Q-Learning** — Two independent Q-tables with decoupled selection and evaluation. Epsilon decay 0.8 → 0.01 with best-Q-table checkpointing to guard against late-episode instability. Final eval: **410.56**.

**Experience Replay** — Replay buffer (capacity 10,000) with random mini-batch updates of size 32. Epsilon decay 0.1 → 0.01. Final eval: **374.64**.

### Achievement Summary

| Component | Status | Eval Score | Points |
|-----------|--------|-----------|--------|
| Monte Carlo | ✅ Passed | 256.64 | 30/30 |
| Q-Learning | ✅ Passed | 443.67 | 30/30 |
| Experiments | ✅ Partial | — | 18.8/25 |
| SARSA (bonus) | ✅ Passed | 454.46 | 5/5 |
| Double Q-Learning (bonus) | ✅ Passed | 410.56 | 7/7 |
| Experience Replay (bonus) | ✅ Passed | 374.64 | 8/8 |
| **Core Total** | | | **78.8/85** |
| **Bonus Total** | | | **20/20** |
| **Grand Total** | | | **98.8/85** |

---

## 3. File Structure and Responsibilities

| File | Purpose | Run Command |
|------|---------|-------------|
| `user_code.py` | **Core submission** — contains Monte Carlo and Q-Learning implementations, all shared helper functions (`discretize_state`, `extract_position`, `format_action`, `choose_action`, `evaluate_policy`), and the `main()` entry point that trains and compares both algorithms | `python user_code.py` |
| `bonus_challenges.py` | **Bonus submission** — SARSA, Double Q-Learning, and Experience Replay implementations, each with their own helper copies and a standalone `evaluate_bonus_challenges()` runner | `python bonus_challenges.py` |
| `evaluate_submission.py` | **Grader script** (provided, not modified) — imports `run_monte_carlo` and `run_q_learning` from `user_code.py`, runs them, measures convergence episode, eval score, and prints the final grade breakdown | `python evaluate_submission.py` |
| `visualize.py` | **Visualization helper** — trains both algorithms headlessly, then re-runs the learned policies with `gui=True` so the PyBullet window opens; window stays open until manually closed | `python visualize.py` |
| `requirements.txt` | Lists all dependencies including `gym-pybullet-drones` (editable install from local clone) | `pip install -r requirements.txt` |
| `report.md` | This report | — |
| `imgs/` | Screenshot of the HoverAviary environment used in the LaTeX report | — |

### How the Files Connect

```
evaluate_submission.py
    └── imports run_monte_carlo(), run_q_learning()  ←── user_code.py
                                                              └── shared helpers used by all algorithms

bonus_challenges.py
    └── self-contained (own helper copies)
    └── evaluate_bonus_challenges() runs all 3 bonus tasks

visualize.py
    └── imports run_monte_carlo(), run_q_learning()  ←── user_code.py
    └── re-runs best policy with gui=True
```

---

## 4. Environment and Problem Setup

The task is to train a single quadrotor in `HoverAviary` (from `gym-pybullet-drones`) to hover at position `[0, 0, 1]`. The drone gets a dense reward signal at every step based on proximity to the target. Episodes run for at most 240 steps (8 seconds at 30 Hz).

### State Space

Only the drone's `(x, y, z)` position relative to the target is used. Continuous coordinates are discretized into uniform bins:

| Axis | Range | Bins | Bin width |
|------|-------|------|-----------|
| x | `[-1, 1]` m | 10 | 0.2 m |
| y | `[-1, 1]` m | 10 | 0.2 m |
| z | `[0, 2]` m | 10 | 0.2 m |

Total state space: `10 × 10 × 10 = 1,000` states.

### Action Space

| Index | Value | Effect |
|-------|-------|--------|
| 0 | −1.0 | Descend |
| 1 | 0.0 | Hold |
| 2 | +1.0 | Ascend |

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NUM_BINS` | 10 | Enough granularity to distinguish positions near the target |
| `EPSILON` | 0.1 | 10% exploration throughout — prevents premature convergence |
| `GAMMA` | 0.99 | High discount; hover needs long-term value, not just immediate reward |
| `ALPHA` | 0.1 | Conservative learning rate; avoids Q-value oscillation |
| `NUM_EPISODES` | 500 | Sufficient for convergence on 1,000 states |
| `MAX_STEPS` | 240 | 8 seconds per episode |

---

## 5. Algorithm Details

### Monte Carlo Control

Generates one full episode, then updates Q-values using actual discounted returns — no bootstrapping.

```
For each episode:
  1. Run episode with ε-greedy policy → collect (s, a, r) at every step
  2. Walk trajectory backwards → G = r + γ·G
  3. For each first-visit (s, a):
       Q(s,a) ← Q(s,a) + α·(G − Q(s,a))
```

### Q-Learning

Updates the Q-table after every single step:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

The behaviour policy is ε-greedy but the target is always greedy (`max`), making it off-policy.

---

## 6. Errors Encountered and How They Were Fixed

This section documents every real bug and environment issue hit during development, in the order they appeared.

---

### Error 1 — `ModuleNotFoundError: No module named 'gym_pybullet_drones'`

**When it appeared:**
```
(.venv) D:\codebase\AR525-RL\RL-AR525\a2> python user_code.py
ModuleNotFoundError: No module named 'gym_pybullet_drones'
```

**Root cause:**
The `gym-pybullet-drones` package was never installed in the active virtual environment. The `requirements.txt` listed `pip install -e ../gym-pybullet-drones`, which is a relative path install — this only works if the package folder exists at that relative location *and* you run the command from the right directory.

**What failed first:**
```bash
pip install -e ../gym-pybullet-drones   # relative path not found
pip install gym-pybullet-drones         # not on PyPI — no matching distribution
```

**Investigation:**
Checked the directory structure and found that the `gym-pybullet-drones` folder existed at `RL-AR525/gym-pybullet-drones/` but was completely **empty** — the git clone had been attempted but never completed.

**Fix:**
```bash
cd D:/codebase/AR525-RL/RL-AR525
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e "D:/codebase/AR525-RL/RL-AR525/gym-pybullet-drones"
```
Used the full absolute path to avoid relative path resolution issues on Windows.

---

### Error 2 — Bonus Challenge 1 (SARSA): Score 0/5 — Evaluation reward ~255, never reaches 300

**Observed output:**
```
SARSA Episode 500/500, Avg Reward: 212.22
SARSA Evaluation: 255.58 (+/- 0.00)
Bonus Points: 0/5
```

**Root cause:**
The SARSA algorithm itself was correct, but it used a **fixed epsilon of 0.1 throughout all 500 episodes**. This means that even at episode 500 — when the Q-table has converged to a good policy — the agent was still acting randomly 10% of the time during training. The Q-table therefore reflected the value of an *exploratory* policy, not a greedy one.

When `evaluate_policy()` then ran the policy **greedily** (ε=0), the Q-table was slightly miscalibrated for pure greedy behavior, and the drone failed to reach the threshold of 300.

Additionally, with fixed epsilon the agent never fully commits to the good policy it discovers in the later episodes — it keeps second-guessing itself with random actions that drag down the late-training rewards.

**Diagnosis:**
The same code on a Linux machine (different PyBullet build) passed, because the physics engine's slightly different numerical behavior allowed the Q-table to converge just above 300 by luck. On Windows with the newer PyBullet build (Feb 2026), convergence was marginally slower and the threshold was missed.

**Fix:**
Added **epsilon decay** — epsilon starts at 0.1 and linearly decays to 0.01 by the final episode:
```python
eps = max(0.01, epsilon * (1.0 - episode / num_episodes))
```
By episode 450+, the agent is 99% greedy during training, so the Q-table accurately reflects the greedy policy. The gap between training behavior and evaluation is eliminated.

**Result after fix:** SARSA Evaluation ≥ 300. **5/5 points.**

---

### Error 3 — Bonus Challenge 3 (Experience Replay): Score 0/8 — Training reward ~315 but evaluation only 189

**Observed output:**
```
Experience Replay Episode 50/500,  Avg Reward: 336.04
Experience Replay Episode 500/500, Avg Reward: 315.77
Experience Replay Evaluation: 189.81 (+/- 0.00)   ← huge gap
Bonus Points: 0/8
```

**Root cause:**
Same root cause as SARSA — fixed epsilon of 0.1 throughout training. With Experience Replay, this problem is worse because:

1. The replay buffer stores experiences from **all** past episodes, including early random ones. Mini-batch updates keep revisiting these old random-action experiences, which continuously "pull" the Q-table toward values that are good for random-ish behavior.
2. The training `total_reward` is measured with epsilon-greedy actions (some random), so it looks high (~315) because lucky random actions sometimes accidentally hover well.
3. But `evaluate_policy()` uses pure greedy — if the Q-table was calibrated for 10% random behavior, greedy evaluation performs much worse (189).

This created a deceptive situation: training looked like it was working (315 reward) while the actual learned greedy policy was poor (189 reward).

**Fix:**
Same epsilon decay as SARSA:
```python
eps = max(0.01, epsilon * (1.0 - episode / num_episodes))
```
By the final episodes, training is nearly greedy. The Q-table converges to a policy that also works under greedy evaluation. The buffer's old random experiences are increasingly outweighed by the larger number of good exploitation experiences added later.

**Result after fix:** Experience Replay Evaluation ≥ 300. **8/8 points.**

---

### Error 4 — Bonus Challenge 2 (Double Q-Learning): Score 0/7 — Every episode exactly 255.58, no learning at all

**Observed output:**
```
Double Q-Learning Episode  50/500, Avg Reward: 255.58
Double Q-Learning Episode 100/500, Avg Reward: 255.58
Double Q-Learning Episode 150/500, Avg Reward: 255.58
...
Double Q-Learning Episode 500/500, Avg Reward: 255.58
Double Q-Learning Evaluation: 255.58 (+/- 0.00)
Bonus Points: 0/7
```

**Root cause — Missing epsilon-greedy (critical bug):**
The original action selection code was:
```python
action = np.argmax(q1[state] + q2[state])
```
This is **pure greedy with no exploration**. With both Q-tables initialized to all zeros, `np.argmax([0, 0, 0])` always returns **index 0** (NumPy's argmax breaks ties by returning the first index). Index 0 maps to action value `-1.0` (descend). So in every single episode, the drone *always descends*, settles at a stable low hover position, and receives reward 255.58. No exploration ever happens. No learning ever happens. All 500 episodes are identical.

Even after applying epsilon decay (ε starting at 0.1), the problem persisted because:
- With ε=0.1, 90% of the time the agent takes the greedy action = action 0 (descend)
- Action 0 gets reinforced with reward 255.58 → Q-table confirms action 0 as best
- The 10% random exploration was not frequent enough to discover that action 2 (ascend to z=1) gives higher reward
- The drone was stuck in a locally stable bad policy from episode 1

**Why 255.58 specifically?**
The value 255.58 is the total reward for an agent that immediately descends to a low hover altitude and stays there for the full 240 steps. It is a stable fixed point — the drone can hover at the wrong altitude indefinitely. The algorithm had no reason to try anything else once it settled here.

**Why it worked on the other system despite the same bug:**
It didn't. Harsh's Linux output also showed `255.58` flat for all 500 Double Q-Learning episodes. Challenge 2 scored 0/7 on both systems. Only SARSA and Experience Replay passed on Linux.

**Fix — High initial epsilon with decay:**
```python
eps = max(0.01, 0.8 * (1.0 - episode / num_episodes))
```
Starting at ε=0.8 means the first 250 episodes are 20–80% random exploration. The drone tries all three actions in many different states, discovers that action 2 (ascend) leads to the target position, and builds up Q-values that correctly reflect this. By episode 400 (ε≈0.24) it begins exploiting the good policy. By episode 500 (ε≈0.008) it is nearly deterministic.

Why 0.8 specifically rather than 0.3 or 0.5? Because the bad local minimum (action 0 = descend) is very stable. Once any Q-values slightly favour action 0, greedy selection reinforces it further. A high initial epsilon is needed to spend enough early episodes exploring before the Q-table has time to lock onto the wrong action.

**Result after fix:** Double Q-Learning Evaluation ≥ 300. **7/7 points.**

---

### Summary of All Fixes

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `ModuleNotFoundError` | Empty git clone, wrong install path | Full path `pip install -e` after re-cloning |
| SARSA: eval < 300 | Fixed ε=0.1 at eval time, Q-table calibrated for exploratory policy | Epsilon decay 0.1 → 0.01 |
| Experience Replay: train 315 but eval 189 | Same fixed ε issue; old random experiences in buffer kept pulling Q-table away from greedy policy | Epsilon decay 0.1 → 0.01 |
| Double Q-Learning: stuck at 255.58 | Missing epsilon-greedy; `argmax(zeros)=0` always descends, never learns | Epsilon decay **0.8 → 0.01** (needed high start to escape local minimum) |

---

## 7. Bonus Challenge Deep Dive

### SARSA vs Q-Learning

The update rule difference is one line:
```
Q-Learning:  td_target = r + γ · max_a' Q(s', a')      ← greedy, off-policy
SARSA:       td_target = r + γ · Q(s', a')              ← a' is actual epsilon-greedy pick
```
SARSA is more conservative — it accounts for the fact that the actual policy might take a bad exploratory action, so it doesn't overvalue states near risky edges. Q-Learning is more optimistic (ignores exploration noise in its target), which makes it faster but sometimes overestimates.

### Double Q-Learning — Why Two Tables?

Standard Q-Learning systematically overestimates because `max Q(s',a')` preferentially selects whichever action has the highest noise at that moment. Two independent tables fix this by decoupling which table picks the action from which table evaluates it:

```
If updating Q1:
    best_next = argmax Q2(s')       ← Q2 selects
    target    = r + γ · Q1(s', best_next)  ← Q1 evaluates
```

A spuriously high value in Q2 will select that action, but Q1 (independently noisy) won't systematically agree — so the overestimation cancels out in expectation.

### Experience Replay — The Training/Eval Gap Explained

Without epsilon decay, here's what happens:
- Training reward (ε=0.1): 315 — drone mostly follows good policy + some lucky random actions
- Eval reward (ε=0.0): 189 — pure greedy hits states where Q-table was calibrated for 10% random noise

With epsilon decay to 0.01:
- Final training episodes are nearly greedy → Q-table reflects pure greedy behavior
- Eval reward matches training reward → both above 300

---

## 8. Grader Evaluation Output

### Core Tasks — `evaluate_submission.py`

```
============================================================
EVALUATING MONTE CARLO IMPLEMENTATION
============================================================
MC Episode  50/500, Avg Reward (last 50): 194.05
MC Episode 100/500, Avg Reward (last 50): 230.17
MC Episode 150/500, Avg Reward (last 50): 128.98
MC Episode 200/500, Avg Reward (last 50): 188.28
MC Episode 250/500, Avg Reward (last 50): 239.75
MC Episode 300/500, Avg Reward (last 50): 261.17
MC Episode 350/500, Avg Reward (last 50): 274.30
MC Episode 400/500, Avg Reward (last 50): 301.12
MC Episode 450/500, Avg Reward (last 50): 274.79
MC Episode 500/500, Avg Reward (last 50): 235.58
Final Evaluation Reward: 256.64 (+/- 0.00)
Final 50-Episode Average: 235.58
Convergence Episode: 358
Q-Table Shape: (10, 10, 10, 3)

============================================================
EVALUATING TD (Q-LEARNING) IMPLEMENTATION
============================================================
Q-Learning Episode  50/500, Avg Reward (last 50): 256.52
Q-Learning Episode 100/500, Avg Reward (last 50): 236.22
Q-Learning Episode 150/500, Avg Reward (last 50): 257.33
Q-Learning Episode 200/500, Avg Reward (last 50): 234.12
Q-Learning Episode 250/500, Avg Reward (last 50): 248.18
Q-Learning Episode 300/500, Avg Reward (last 50): 277.26
Q-Learning Episode 350/500, Avg Reward (last 50): 321.08
Q-Learning Episode 400/500, Avg Reward (last 50): 332.19
Q-Learning Episode 450/500, Avg Reward (last 50): 385.06
Q-Learning Episode 500/500, Avg Reward (last 50): 386.41
Final Evaluation Reward: 443.67 (+/- 0.00)
Final 50-Episode Average: 386.41
Convergence Episode: 434
Q-Table Shape: (10, 10, 10, 3)

============================================================
FINAL GRADE
============================================================
Feedback:
  ✓ Monte Carlo implementation PASSED
  ✓ TD (Q-Learning) implementation PASSED
  ~ One algorithm converged reasonably

Score Breakdown:
  Monte Carlo: 30.0/30
  TD Learning: 30.0/30
  Experiments: 18.8/25
  -------------------
  TOTAL: 78.8/85
```

### Analysis of Grader Results

**Monte Carlo (30/30):** Passed with eval score 256.64. The training curve was volatile — reward dipped to 128 at episode 150 before recovering to 301 at episode 400, then fell back to 235 by episode 500. This is characteristic MC behaviour: high variance from full-episode returns means the final snapshot isn't always the best policy seen. Convergence was detected at episode 358. The 256 eval score is lower than Q-Learning because MC's unbiased-but-noisy updates leave the Q-table in a more uncertain state at episode 500.

**Q-Learning (30/30):** Passed with eval score 443.67. Clear steady improvement from 256 at episode 50 to 386 at episode 500 — exactly what TD's step-wise updates enable. Convergence at episode 434 (later than MC's 358, but the final policy is far stronger). The 443 eval score vs MC's 256 confirms Q-Learning's advantage on this dense-reward, long-episode task.

**Experiments (18.8/25):** The partial deduction (`~ One algorithm converged reasonably` rather than both) is because MC's final 50-episode average (235.58) was below the grader's "fully converged" threshold, even though it passed the binary PASSED check. Q-Learning's 386 final average satisfied the convergence criterion; MC did not due to its high variance late in training.

### Score Summary

| Component | Score | Notes |
|-----------|-------|-------|
| Monte Carlo | 30.0 / 30 | Passed binary check; eval 256.64 |
| TD Learning | 30.0 / 30 | Passed; eval 443.67, strong convergence |
| Experiments | 18.8 / 25 | Deducted: only Q-Learning fully converged |
| **Core Subtotal** | **78.8 / 85** | |
| SARSA (bonus) | 5 / 5 | Eval 454.46 |
| Double Q-Learning (bonus) | 7 / 7 | Eval 410.56 |
| Experience Replay (bonus) | 8 / 8 | Eval 374.64 |
| **Bonus Subtotal** | **20 / 20** | |
| **Grand Total** | **98.8 / 85** | Exceeds full marks with bonus |

---

### Post-Merge Run — `user_code.py` (NUM_BINS=8, ALPHA=0.2, Epsilon Decay)

After merging `user_code1.py` improvements into `user_code.py` — reducing `NUM_BINS` from 10 to 8 and increasing `ALPHA` from 0.1 to 0.2 — the algorithms were re-run directly via `python user_code.py`:

```
============================================================
RL Assignment: Monte Carlo vs TD Learning
Task: Drone Hover at z=1.0
============================================================
Environment: HoverAviary
Target Position: [0 0 1]
Episode Length: 240 steps (8.0 seconds)

----------------------------------------
Training Monte Carlo...
----------------------------------------
MC Episode  50/500, Avg Reward (last 50): 243.26
MC Episode 100/500, Avg Reward (last 50): 196.38
MC Episode 150/500, Avg Reward (last 50): 267.30
MC Episode 200/500, Avg Reward (last 50): 288.41
MC Episode 250/500, Avg Reward (last 50): 257.16
MC Episode 300/500, Avg Reward (last 50): 232.37
MC Episode 350/500, Avg Reward (last 50): 210.93
MC Episode 400/500, Avg Reward (last 50): 202.51
MC Episode 450/500, Avg Reward (last 50): 168.39
MC Episode 500/500, Avg Reward (last 50): 158.64
MC Final Evaluation: 432.71 (+/- 0.00)

----------------------------------------
Training Q-Learning...
----------------------------------------
Q-Learning Episode  50/500, Avg Reward (last 50): 236.68
Q-Learning Episode 100/500, Avg Reward (last 50): 329.40
Q-Learning Episode 150/500, Avg Reward (last 50): 356.72
Q-Learning Episode 200/500, Avg Reward (last 50): 384.67
Q-Learning Episode 250/500, Avg Reward (last 50): 355.68
Q-Learning Episode 300/500, Avg Reward (last 50): 432.39
Q-Learning Episode 350/500, Avg Reward (last 50): 380.44
Q-Learning Episode 400/500, Avg Reward (last 50): 436.07
Q-Learning Episode 450/500, Avg Reward (last 50): 343.04
Q-Learning Episode 500/500, Avg Reward (last 50): 438.68
TD Final Evaluation: 453.58 (+/- 0.00)

============================================================
RESULTS SUMMARY
============================================================
Monte Carlo - Final Avg Reward (last 50): 158.64
Q-Learning  - Final Avg Reward (last 50): 438.68

Monte Carlo - Evaluation: 432.71 (+/- 0.00)
Q-Learning  - Evaluation: 453.58 (+/- 0.00)

Q-Learning performed better!
```

### Analysis of Post-Merge Results

**Monte Carlo (Eval: 432.71):** A significant jump from the grader run's 256.64. The smaller Q-table (8³=512 states vs 10³=1000) and higher learning rate (α=0.2) accelerate value propagation — the drone reaches a stable greedy policy that the evaluation detects as 432.71. Note that the *training* curve peaks at episode 200 (288) then falls to 158 by episode 500 — this is the typical MC variance pattern where late-episode epsilon decay makes training rewards drop (less lucky exploration) while the Q-table itself has already converged to a strong greedy policy.

**Q-Learning (Eval: 453.58):** Strong and consistent improvement across all 500 episodes — from 236 at episode 50 to 438 at episode 500. The higher α=0.2 accelerates convergence and the smaller state space lets Q-values propagate to neighbours faster. Evaluation score of 453.58 confirms a well-converged greedy policy.

**Comparison with original grader run:**

| Algorithm | Original Eval (NUM_BINS=10, α=0.1) | Post-Merge Eval (NUM_BINS=8, α=0.2) | Improvement |
|-----------|--------------------------------------|--------------------------------------|-------------|
| Monte Carlo | 256.64 | **432.71** | +176 |
| Q-Learning | 443.67 | **453.58** | +10 |

The larger gain in MC confirms that the smaller, denser state space benefits MC more — fewer unseen states means the first-visit updates cover the space more completely within 500 episodes.

---

## 9. Final Results — Bonus Challenges

### Bonus Challenges — Final Output

```
--- Challenge 1: SARSA (5 points) ---
SARSA Episode  50/500, Avg Reward: 280.15
SARSA Episode 100/500, Avg Reward: 312.14
SARSA Episode 150/500, Avg Reward: 311.88
SARSA Episode 200/500, Avg Reward: 337.28
SARSA Episode 250/500, Avg Reward: 368.45
SARSA Episode 300/500, Avg Reward: 359.20
SARSA Episode 350/500, Avg Reward: 348.37
SARSA Episode 400/500, Avg Reward: 392.34
SARSA Episode 450/500, Avg Reward: 404.41
SARSA Episode 500/500, Avg Reward: 451.03
SARSA Evaluation: 454.46 (+/- 0.00)
Bonus Points: 5/5

--- Challenge 2: Double Q-Learning (7 points) ---
Double Q-Learning Episode  50/500, Avg Reward: 325.48
Double Q-Learning Episode 100/500, Avg Reward: 294.27
Double Q-Learning Episode 150/500, Avg Reward: 263.72
Double Q-Learning Episode 200/500, Avg Reward: 289.39
Double Q-Learning Episode 250/500, Avg Reward: 292.68
Double Q-Learning Episode 300/500, Avg Reward: 268.89
Double Q-Learning Episode 350/500, Avg Reward: 264.94
Double Q-Learning Episode 400/500, Avg Reward: 248.58
Double Q-Learning Episode 450/500, Avg Reward: 343.67
Double Q-Learning Episode 500/500, Avg Reward: 308.06
Double Q-Learning Evaluation: 410.56 (+/- 0.00)
Bonus Points: 7/7

--- Challenge 3: Experience Replay (8 points) ---
Experience Replay Episode  50/500, Avg Reward: 296.44
Experience Replay Episode 100/500, Avg Reward: 336.79
Experience Replay Episode 150/500, Avg Reward: 269.52
Experience Replay Episode 200/500, Avg Reward: 266.26
Experience Replay Episode 250/500, Avg Reward: 316.94
Experience Replay Episode 300/500, Avg Reward: 299.76
Experience Replay Episode 350/500, Avg Reward: 280.40
Experience Replay Episode 400/500, Avg Reward: 300.34
Experience Replay Episode 450/500, Avg Reward: 329.79
Experience Replay Episode 500/500, Avg Reward: 285.16
Experience Replay Evaluation: 374.64 (+/- 0.00)
Bonus Points: 8/8
```

### Score Summary

| Algorithm | Eval Score | Threshold | Points |
|-----------|-----------|-----------|--------|
| SARSA | 454.46 | ≥ 300 | **5/5** |
| Double Q-Learning | 410.56 | ≥ 300 | **7/7** |
| Experience Replay | 374.64 | ≥ 300 | **8/8** |
| **Total Bonus** | | | **20/20** |

All three algorithms scored well above the 300 threshold. SARSA reached the highest evaluation score (454.46), showing that on-policy TD with proper epsilon decay and best-checkpoint tracking produces a very stable greedy policy. Double Q-Learning (410.56) benefited most from the checkpoint fix — its training curve was volatile (dipping to 248 at episode 400) but the saved best snapshot from an earlier peak was correctly returned at evaluation. Experience Replay (374.64) was the most consistent across runs, with training rewards staying broadly in the 266–336 range throughout.

### Learning Curve Observations

**SARSA** showed the clearest improvement trend — reward climbed steadily from 280 at episode 50 to 451 at episode 500. This is consistent with epsilon decay: as the agent becomes more deterministic over time, each episode more closely resembles the greedy policy that evaluation measures.

**Double Q-Learning** had the most volatile training curve — rewards fluctuated between 248 and 343 in episodes 100–450. This is expected: with ε=0.8 starting exploration, the agent spends the first half of training sampling broadly and the Q-values are noisy. The best-checkpoint mechanism rescued a peak at episode 50 (325) and episode 450 (343) to deliver a 410 evaluation score despite the noise.

**Experience Replay** was stable throughout but showed no clear upward trend after episode 100 (336). The replay buffer helps avoid catastrophic forgetting but doesn't accelerate final convergence beyond what the buffer diversity supports.

---

## 10. Algorithm Comparison

### MC vs. Q-Learning

| Property | Monte Carlo | Q-Learning |
|----------|-------------|------------|
| Updates | End of episode | Every step |
| Convergence speed | Slower | Faster |
| Variance | High (real returns) | Low (bootstrapped) |
| Bias | None | Small (max operator) |
| Policy | On-policy | Off-policy |

For the hover task, Q-Learning is faster because episodes are 240 steps long and the reward is dense. MC waits the full episode before learning anything.

### Bonus Algorithm Comparison

| Algorithm | Eval Score | Key Mechanism | Stability |
|-----------|-----------|---------------|-----------|
| SARSA | 454.46 | On-policy, uses actual next action | High (steady improvement) |
| Double Q-Learning | 410.56 | Decoupled selection + evaluation tables | Medium (volatile, saved by checkpoint) |
| Experience Replay | 374.64 | Random mini-batch from replay buffer | High (consistent across runs) |

---

## 11. Conclusion

Both Monte Carlo and Q-Learning successfully learned a stable drone hover policy. Q-Learning converges faster due to online step-wise updates. Monte Carlo's unbiased returns match Q-Learning quality given enough episodes.

The debugging process revealed an important practical lesson: **the gap between training behavior and evaluation behavior is the most common silent failure mode in RL**. All three bonus challenge failures came down to the same root cause — the policy seen during training (with exploration noise) did not match the policy tested at evaluation (pure greedy). Epsilon decay is the standard fix: it gradually shifts training behavior toward what evaluation will see, so the Q-table converges to a policy that actually works when deployed.

The three bonus algorithms trace the path from tabular RL toward DQN: SARSA adds on-policy conservatism, Double Q-Learning corrects maximization bias, and Experience Replay improves stability through decorrelated updates — all three ideas appear directly in the original DQN paper.
