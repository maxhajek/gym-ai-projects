import json
import os
import datetime
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters & Configurations
CONFIG = {
    "ALPHA": 0.1,
    "GAMMA": 0.99,
    "EPSILON": 0.6,
    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.995,
    "EPISODES": 15000,
    "PENALTY": -375,
    "TARGET_STEPS": 200,
}

env = gym.make("CartPole-v1")
n_actions = env.action_space.n

# Define bins for each observation dimension
CART_POSITION_BINS = np.linspace(-4.8, 4.8, 24)
CART_VELOCITY_BINS = np.linspace(-4, 4, 24)
POLE_ANGLE_BINS = np.linspace(-0.418, 0.418, 24)
POLE_VELOCITY_BINS = np.linspace(-4, 4, 24)


def discretize_state(state):
    cart_position_idx = np.digitize(state[0], CART_POSITION_BINS)
    cart_velocity_idx = np.digitize(state[1], CART_VELOCITY_BINS)
    pole_angle_idx = np.digitize(state[2], POLE_ANGLE_BINS)
    pole_velocity_idx = np.digitize(state[3], POLE_VELOCITY_BINS)
    return cart_position_idx, cart_velocity_idx, pole_angle_idx, pole_velocity_idx


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state])


def update_q_value(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    q_target = reward + CONFIG["GAMMA"] * q_table[next_state][best_next_action]
    q_table[state][action] = (1 - CONFIG["ALPHA"]) * q_table[state][action] + CONFIG[
        "ALPHA"
    ] * q_target


def train():
    previousCnt = []
    metrics = {"ep": [], "avg": [], "min": [], "max": []}
    epsilon = CONFIG["EPSILON"]

    for episode in range(CONFIG["EPISODES"]):
        continuous_state, _ = env.reset(seed=42)
        state = discretize_state(continuous_state)
        done = False
        cnt = 0
        while not done:
            action = choose_action(state, epsilon)
            cnt += 1
            next_continuous_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_continuous_state)
            if done and cnt < CONFIG["TARGET_STEPS"]:
                reward = CONFIG["PENALTY"]
            update_q_value(state, action, reward, next_state)
            state = next_state
        previousCnt.append(cnt)
        epsilon *= CONFIG["EPSILON_DECAY"]
        epsilon = max(CONFIG["EPSILON_MIN"], epsilon)
        if episode % 100 == 0:
            log_metrics(episode, previousCnt, metrics)
    env.close()
    return metrics


def log_metrics(episode, previousCnt, metrics):
    latestRuns = previousCnt[-100:]
    averageCnt = sum(latestRuns) / len(latestRuns)
    metrics["ep"].append(episode)
    metrics["avg"].append(averageCnt)
    metrics["min"].append(min(latestRuns))
    metrics["max"].append(max(latestRuns))
    print(
        f"Run: {episode}, Average: {averageCnt}, Min: {min(latestRuns)}, Max: {max(latestRuns)}"
    )


def plot_metrics(metrics):
    plt.plot(metrics["ep"], metrics["avg"], label="average rewards")
    plt.plot(metrics["ep"], metrics["min"], label="min rewards")
    plt.plot(metrics["ep"], metrics["max"], label="max rewards")
    plt.legend(loc=4)
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f"pole-cart-{t}.png")


def save_q_table():
    with open("q_table.json", "w") as f:
        json.dump(q_table.tolist(), f, indent=1)


def load_q_table():
    global q_table
    if os.path.exists("q_table.json"):
        with open("q_table.json", "r") as f:
            q_table = np.array(json.load(f))
    else:
        q_table = np.zeros((25, 25, 25, 25, n_actions))


if __name__ == "__main__":
    load_q_table()
    metrics = train()
    plot_metrics(metrics)
    save_q_table()
