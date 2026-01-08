# training/train_agent.py
import os
import sys
# Ensure project root is on path when running as script (safe-guard)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import datetime
import tensorflow as tf
from config import Config as cfg
from env.simulation_env import GridEnvironment
from agents.replay_buffer import SequenceReplayBuffer
from agents.dqn_agent import LSTM_DQN_Agent
from utils.metrics import RunningStats

# reproducibility seeds
np.random.seed(cfg.SEED)
tf.random.set_seed(cfg.SEED)

# create dirs
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

# env & agent
env = GridEnvironment(**cfg.ENV)
agent = LSTM_DQN_Agent(input_shape=(cfg.TIME_STEPS,
                                    cfg.ENV['vision_size'],
                                    cfg.ENV['vision_size'], 3),
                       n_actions=5,
                       lr=cfg.LR,
                       gamma=cfg.GAMMA,
                       lstm_units=cfg.LSTM_UNITS,
                       soft_update=cfg.SOFT_UPDATE,
                       tau=cfg.TAU,
                       grad_clip=10.0)

buffer = SequenceReplayBuffer(capacity=cfg.BUFFER_CAPACITY)
stats = RunningStats(window=100)

# TensorBoard
log_dir = os.path.join(cfg.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(log_dir)

# initialize replay buffer with random policy
print("Initializing replay buffer with random actions...")
obs = env.reset()
frames = [obs for _ in range(cfg.TIME_STEPS)]
for i in range(cfg.MIN_REPLAY_SIZE):
    action = np.random.randint(0, 5)
    next_obs, reward, done, _ = env.step(action)
    next_frames = frames[1:] + [next_obs]
    buffer.push(frames, action, reward, next_frames, float(done))
    if done:
        obs = env.reset()
        frames = [obs for _ in range(cfg.TIME_STEPS)]
    else:
        frames = next_frames
if len(buffer) < cfg.MIN_REPLAY_SIZE:
    raise RuntimeError("Replay buffer initialization failed.")

print(f"Replay buffer initialized: {len(buffer)} transitions.")

# training loop
epsilon = cfg.EPS_START
global_step = 0

for ep in range(1, cfg.EPISODES + 1):
    obs = env.reset()
    frames = [obs for _ in range(cfg.TIME_STEPS)]
    ep_reward = 0.0
    done = False

    while not done:
        action = agent.act(np.array(frames), epsilon)
        next_obs, reward, done, _ = env.step(action)
        next_frames = frames[1:] + [next_obs]
        buffer.push(frames, action, reward, next_frames, float(done))
        frames = next_frames
        ep_reward += reward
        global_step += 1

        # Training step
        if len(buffer) >= cfg.BATCH_SIZE and global_step >= cfg.LEARNING_STARTS:
            obs_b, act_b, rew_b, next_b, done_b = buffer.sample(cfg.BATCH_SIZE)
            # debug checks (first few batches)
            if global_step < 1000 and (global_step % 200 == 0):
                print("DEBUG SAMPLE SHAPES:", obs_b.shape, act_b.shape, rew_b.shape, next_b.shape, done_b.shape)
                print("dones unique:", np.unique(done_b))
            loss = agent.train_on_batch(obs_b.astype(np.float32),
                                        act_b.astype(np.int32),
                                        rew_b.astype(np.float32),
                                        next_b.astype(np.float32),
                                        done_b.astype(np.float32))
            # soft update or periodic hard update
            if cfg.SOFT_UPDATE:
                agent.update_target_network(hard=False)
            elif global_step % (cfg.TARGET_UPDATE_EVERY * env.max_steps) == 0:
                agent.update_target_network(hard=True)

            # log loss occasionally
            if global_step % 100 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("Loss", loss, step=global_step)

        # safety break if too long (shouldn't happen)
        if ep_reward < -1e6:
            print("Abnormal reward, breaking.")
            done = True

    # end episode
    stats.add(ep_reward)
    # decay epsilon
    epsilon = max(cfg.EPS_END, epsilon * cfg.EPS_DECAY)

    # log per episode
    with summary_writer.as_default():
        tf.summary.scalar("Episode Reward", ep_reward, step=ep)
        tf.summary.scalar("Average Reward", stats.mean(), step=ep)
        tf.summary.scalar("Epsilon", epsilon, step=ep)
        tf.summary.scalar("Buffer Size", len(buffer), step=ep)

    if ep % 10 == 0:
        print(f"Episode {ep} | Reward: {ep_reward:.3f} | AvgReward(last{stats.window}): {stats.mean():.3f} | Epsilon: {epsilon:.3f} | Buffer: {len(buffer)}")

    # periodic save
    if ep % 100 == 0:
        agent.save(os.path.join(cfg.SAVE_DIR, f"checkpoint_ep{ep}"))

# final save
agent.save(cfg.SAVE_DIR)
print("Training complete.")
print(f"TensorBoard logs at: {log_dir}")
print("Run: tensorboard --logdir", cfg.LOG_DIR)
