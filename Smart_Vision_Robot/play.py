# play.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from config import Config as cfg
from env.simulation_env import GridEnvironment
from agents.dqn_agent import LSTM_DQN_Agent

env = GridEnvironment(**cfg.ENV)
agent = LSTM_DQN_Agent(input_shape=(cfg.TIME_STEPS,
                                    cfg.ENV['vision_size'],
                                    cfg.ENV['vision_size'], 3),
                       n_actions=5,
                       lr=cfg.LR,
                       gamma=cfg.GAMMA,
                       lstm_units=cfg.LSTM_UNITS,
                       soft_update=cfg.SOFT_UPDATE,
                       tau=cfg.TAU)

agent.load(cfg.SAVE_DIR)

obs = env.reset()
frames = [obs for _ in range(cfg.TIME_STEPS)]
done = False
total_reward = 0.0
while not done:
    action = agent.act(np.array(frames), epsilon=0.0)
    next_obs, reward, done, _ = env.step(action)
    frames = frames[1:] + [next_obs]
    total_reward += reward
    env.render(scale=2)

print("Play finished. Total reward:", total_reward)
