#!/usr/bin/env python3
"""
evaluate_save_videos.py

Evaluates a saved model over multiple episodes and saves videos:
  - successes -> videos/success/
  - failures  -> videos/failure/

Usage:
  python evaluate_save_videos.py --model models --episodes 50 --fps 8
"""

import os
import argparse
import datetime
import numpy as np
import cv2
from config import Config as cfg
from env.simulation_env import GridEnvironment
from agents.dqn_agent import LSTM_DQN_Agent

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def overlay_text(img, text, pos=(10,30), font_scale=0.8, color=(255,255,255), thickness=2, bg_color=(0,0,0)):
    """
    Draw semi-opaque background box and text on img (RGB).
    """
    # convert to BGR for cv2.putText if needed later; here we assume input is RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    # text size
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    # background rectangle with some transparency
    cv2.rectangle(bgr, (x-6, y-h-6), (x + w + 6, y + 6), bg_color, -1)
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    # convert back to RGB for uniformity in rest of code
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def frames_to_video(frames, out_path, fps=8):
    """
    frames: list of RGB uint8 images (H,W,3)
    writes MP4 using OpenCV (mp4v)
    """
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    # ensure uint8 & consistent sizes
    frames = [np.asarray(f).astype(np.uint8) for f in frames]
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")
    for f in frames:
        # convert RGB->BGR for writer
        if f.shape[2] == 3:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        else:
            bgr = f
        writer.write(bgr)
    writer.release()

def load_agent(model_input_dir_or_file):
    # Accept either folder containing lstm_dqn_model.keras or a full path to that file
    if os.path.isdir(model_input_dir_or_file):
        model_dir = model_input_dir_or_file
    else:
        # if a file path was given, use its directory
        model_dir = os.path.dirname(model_input_dir_or_file) or "."
    agent = LSTM_DQN_Agent(input_shape=(cfg.TIME_STEPS,
                                       cfg.ENV['vision_size'],
                                       cfg.ENV['vision_size'], 3),
                           n_actions=5,
                           lr=cfg.LR,
                           gamma=cfg.GAMMA,
                           lstm_units=cfg.LSTM_UNITS,
                           soft_update=cfg.SOFT_UPDATE,
                           tau=cfg.TAU)
    agent.load(model_dir)
    return agent

def main(model_path, episodes=20, fps=8, save_success_only=False, single_folder=False, render_scale=2):
    env = GridEnvironment(**cfg.ENV)
    agent = load_agent(model_path)

    # prepare folders
    base_dir = "videos"
    if single_folder:
        ensure_dir(base_dir)
    else:
        success_dir = os.path.join(base_dir, "success")
        fail_dir = os.path.join(base_dir, "failure")
        ensure_dir(success_dir)
        ensure_dir(fail_dir)

    saved = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        frames_seq = [obs for _ in range(cfg.TIME_STEPS)]
        done = False
        total_reward = 0.0
        frame_buffer = []

        # initial full-grid render if available
        try:
            img0 = env._render_grid_image()
        except Exception:
            img0 = (obs * 255).astype(np.uint8)
            img0 = cv2.resize(img0, (env.W * 20, env.H * 20), interpolation=cv2.INTER_NEAREST)

        # convert to RGB (grid render uses BGR in env._render_grid_image())
        # we will assume env._render_grid_image returns BGR (OpenCV), so convert to RGB for overlay consistency
        if img0.dtype != np.uint8:
            img0 = (img0 * 255).astype(np.uint8)
        try:
            # try detect if image is grayscale/patch size
            if img0.ndim == 2:
                img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            else:
                # if env._render_grid_image created BGR, convert to RGB
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        except Exception:
            # fallback: if conversion fails, just normalize
            pass

        # overlay initial text and append
        tv = overlay_text(img0, f"Ep {ep} | Reward: {total_reward:.2f}", pos=(10,30))
        if render_scale != 1:
            tv = cv2.resize(tv, (tv.shape[1] * render_scale, tv.shape[0] * render_scale), interpolation=cv2.INTER_NEAREST)
        frame_buffer.append(tv)

        while not done:
            action = agent.act(np.array(frames_seq), epsilon=0.0)
            next_obs, reward, done, _ = env.step(action)
            frames_seq = frames_seq[1:] + [next_obs]
            total_reward += reward

            # render full grid if available, else upscale observation
            try:
                img = env._render_grid_image()
            except Exception:
                img = (next_obs * 255).astype(np.uint8)
                img = cv2.resize(img, (env.W * 20, env.H * 20), interpolation=cv2.INTER_NEAREST)

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

            # overlay ep & reward on the RGB image
            txt = f"Ep {ep} | Reward: {total_reward:.2f}"
            img_with_text = overlay_text(img, txt, pos=(10, 30))

            if render_scale != 1:
                img_with_text = cv2.resize(img_with_text, (img_with_text.shape[1] * render_scale, img_with_text.shape[0] * render_scale),
                                           interpolation=cv2.INTER_NEAREST)
            frame_buffer.append(img_with_text)

        # decide destination and name
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if single_folder:
            out_dir = base_dir
        else:
            out_dir = success_dir if total_reward > 0 else fail_dir

        # if saving only successes, skip failures
        if save_success_only and total_reward <= 0:
            print(f"[Eval] Ep {ep} | Reward {total_reward:.3f} -> NOT saved (only successes).")
            continue

        fname = f"ep{ep}_{'succ' if total_reward>0 else 'fail'}_{ts}_r{total_reward:.2f}.mp4"
        out_path = os.path.join(out_dir, fname)
        try:
            frames_to_video(frame_buffer, out_path, fps=fps)
            saved.append(out_path)
            print(f"[Eval] Ep {ep} | Reward {total_reward:.3f} -> Saved: {out_path}")
        except Exception as e:
            print(f"[Eval] Ep {ep} | Reward {total_reward:.3f} -> Failed to save video: {e}")

    print("Evaluation done. Saved videos:")
    for p in saved:
        print("  ", p)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model folder (containing lstm_dqn_model.keras) or full file path")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--save_success_only", action="store_true", help="Save only successful episodes (default: save both)")
    p.add_argument("--single_folder", action="store_true", help="Save all videos in single folder ./videos/")
    p.add_argument("--render_scale", type=int, default=2, help="Scale factor for rendered frames")
    args = p.parse_args()

    main(args.model, episodes=args.episodes, fps=args.fps,
         save_success_only=args.save_success_only, single_folder=args.single_folder,
         render_scale=args.render_scale)
