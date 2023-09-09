import json
import os
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

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


def run():
    continuous_state, _ = env.reset(seed=42)
    state = discretize_state(continuous_state)
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_continuous_state, reward, done, _, _ = env.step(action)
        state = discretize_state(next_continuous_state)
    env.close()


def load_q_table():
    global q_table
    if os.path.exists("q_table.json"):
        with open("q_table.json", "r") as f:
            q_table = np.array(json.load(f))
    else:
        raise Exception("q_table.json not found")


if __name__ == "__main__":
    load_q_table()
    run()
