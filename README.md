# Overcoming Reward Shaping Bottlenecks in Robotic Arm Manipulation using SAC and HER

This project explores continuous robotic arm manipulation (grasping, lifting, and goal placement) using Reinforcement Learning. It includes two implementations:
1. Standard **Soft Actor-Critic (SAC)** with reward shaping on the Robosuite `Panda` Lift task.
2. Goal-conditioned **SAC with Hindsight Experience Replay (HER)** on the Gymnasium-Robotics `FetchPickAndPlace-v4` environment with sparse rewards.

---

## Motivation: Transitioning from Reward Shaping to HER

### The Hovering Issue with Reward Shaping
Initially, I implemented standard SAC on the Robosuite `Panda` arm environment (`Lift`) using a custom reward shaping function (`CustomLift`). The shaped reward function combined three terms:
1. **Reaching reward**
2. **Grasp bonus**
3. **Lift reward**

While reward shaping was intended to guide the policy toward lifting the object, it still suffered from a severe local optimum: **after grasping the object, the agent learned to hover around it or hold it in place instead of actually lifting it off the table.**

Because hovering and holding provided a constant stream of positive rewards without the risk of dropping the object or making unstable movements, the agent got stuck in this sub-optimal behavior.


### Fixing the Issue with HER and Sparse Rewards
To solve the hovering problem, I moved from reward shaping to **Goal-Conditioned SAC with Hindsight Experience Replay (HER)** on the `FetchPickAndPlace-v4` environment.

1. **Sparse Rewards**: The environment uses binary sparse rewards ($0$ when the object is at the goal within a threshold, $-1$ otherwise). This removes intermediate shaped reward incentives that caused the hovering behavior.
2. **HER Buffer**: During replay buffer sampling, failed trajectories are replayed by substituting the achieved end-state as the desired goal. This allows the policy to learn reaching, grasping, lifting, and placing behaviors directly from sparse feedback.


---

## Project Structure

```
robot-arm/
├── sac/                        # Robosuite SAC implementation (shaped rewards)
│   ├── custom_lift.py          # CustomLift environment with reward shaping
│   ├── sac_torch.py            # SAC agent implementation with auto-tuned alpha
│   ├── networks.py             # Actor (Gaussian policy) and Twin Q-networks
│   ├── buffer.py               # Standard replay buffer
│   ├── train_sac.py            # Training script for Robosuite Lift
│   └── test.py                 # Deterministic evaluation script
│
├── sac_her/                    # SAC + HER implementation (sparse rewards)
│   ├── her_buffer.py           # HER buffer with future goal resampling & state normalization
│   ├── train_sac_her.py        # Training script for FetchPickAndPlace-v4
│   └── test_sac.py             # Deterministic evaluation script
│
└── requirements.txt            # Python dependencies
```


## Requirements and Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/robot-arm.git
cd robot-arm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies include: `torch`, `gymnasium`, `gymnasium-robotics`, `robosuite`, `numpy`, and `tensorboard`.

---

## How to Run

### 1. Training and Testing SAC + HER (FetchPickAndPlace-v4)

To start training SAC with HER:
```bash
python3 sac_her/train_sac_her.py
```
Checkpoints will be saved in `checkpoints/` and TensorBoard logs are written to `runs/`.

To test a trained checkpoint deterministically:
```bash
python3 sac_her/test_sac.py
```

### 2. Training and Testing Standard SAC (Robosuite Lift)

To train standard SAC with shaped rewards:
```bash
python3 -m sac/train_sac.py
```

To run deterministic evaluation:
```bash
python3 -m sac/test_sac.py
```

---

## Logging and Metrics

To track training progress (Q-values, actor/critic losses, success rate, and temperature parameter $\alpha$), run TensorBoard:

```bash
tensorboard --logdir=runs
# For Robosuite SAC logs:
tensorboard --logdir=logs
```

---

## References

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, 2018.
- Andrychowicz et al., *Hindsight Experience Replay*, 2017.
- [Gymnasium Robotics Documentation](https://robotics.gymnasium.farama.org/)
- [Robosuite Documentation](https://robosuite.ai/)
# robot-manipulation-rl
# robot-manipulation-rl
# robot-manipulation-rl
# robot-manipulation-rl
# robot-manipulation-rl
