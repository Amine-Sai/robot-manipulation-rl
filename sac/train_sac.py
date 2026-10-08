from collections import deque

import numpy as np
import robosuite as suite
from sac.custom_lift import CustomLift
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter

from sac.sac_torch import Agent


ENV_NAME = "Lift"
ROBOT = "Panda"
HORIZON = 500
CONTROL_FREQ = 20

N_EPISODES = 2000
WARMUP_STEPS = 5000

OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "object-state",
]


def make_env(reward_shaping=True, horizon=HORIZON, control_freq=CONTROL_FREQ, **env_kwargs):
    env = CustomLift(
            robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        use_camera_obs=False,
        use_object_obs=True,
        has_offscreen_renderer=False,
        horizon=horizon,
        reward_shaping=reward_shaping,
        control_freq=control_freq,
        **env_kwargs,
    )

    env = GymWrapper(env, keys= OBS_KEYS)

    return env


def train(
    reward_shaping=True,
    n_episodes=N_EPISODES,
    warmup_steps=WARMUP_STEPS,
    load_model=False,
):
    phase = "shaped" if reward_shaping else "sparse"

    env = make_env(reward_shaping)

    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
    )

    if load_model:
        agent.load_models()
        print(f"Loaded model:")

    writer = SummaryWriter(f"logs/{phase}")

    total_steps = 0
    success_count = 0
    best_score = -np.inf
    next_log_step = 5000

    score_history = deque(maxlen=100)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        success = False
        score = 0.0
        grasp = False

        while not done:
            total_steps += 1

            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs)

            next_obs, reward, done, _ = env.step(action)

            success = bool(env.unwrapped._check_success())
            grasp |= bool(env.unwrapped._check_grasp(
                gripper=env.unwrapped.robots[0].gripper,
                 object_geoms=env.unwrapped.cube,
                 )
)

            if success:
                success_count += 1

            agent.remember(
                obs,
                action,
                reward,
                next_obs,
                success,
            )

            obs = next_obs
            score += reward

            if total_steps >= warmup_steps:
                agent.learn()

            if success:
                break

        score_history.append(score)
        average_score = np.mean(score_history)

        writer.add_scalars(
            "Training",
            {
                "Episode Reward": score,
                "Average Reward": average_score,
                "Grasp": int(grasp),
                "Success": int(success),
            },
            episode,
        )

        if total_steps >= warmup_steps and score > best_score:
            agent.save_models()
            best_score = score

        if episode % 10 == 0:
            success_rate = success_count / (episode + 1)

            print(
                f"Ep {episode:4d} | "
                f"Steps {total_steps:7d} | "
                f"Avg {average_score:7.2f} | "
                f"Success {success_rate:.1%}"
            )

        if total_steps >= next_log_step:
            if total_steps >= warmup_steps:
                print(
                    f"Step {total_steps} | "
                    f"Q1 {agent.q1.mean().item():.2f} | "
                    f"Q2 {agent.q2.mean().item():.2f} | "
                    f"LogP {agent.log_probs.mean().item():.2f} | "
                    f"Alpha {agent.alpha:.4f}"
                )

            next_log_step += 5000

    env.close()
    writer.close()

    print(
        f"\nFinished {phase} training | "
        f"Steps: {total_steps} | "
        f"Successes: {success_count}"
    )


def run_experiment():
    train(
        reward_shaping=True,
        load_model=False,
    )


if __name__ == "__main__":
    run_experiment()