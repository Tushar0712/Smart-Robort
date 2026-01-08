# Smart Vision Robot — CNN + LSTM + DQN

Run training:
python training/train_agent.py

Evaluate:
python training/evaluate.py --model models/final_seq_model.keras

Play / demo:
python training/play.py --model models/final_seq_model.keras

# 🤖 Smart Vision Robot: Learning to Navigate Using AI

An autonomous navigation system where an AI agent **learns to navigate complex grid environments** using **Deep Reinforcement Learning (DRL)**.  
It integrates **CNNs (Convolutional Neural Networks)** for visual input processing and **RNNs (LSTM)** for sequential decision-making.

---

## 🎯 Objective
To design an AI system that:
- **Sees** the environment using a camera-like grid processed through CNN layers.  
- **Thinks** using RNN (LSTM) to remember past movements.  
- **Learns** optimal navigation through **Q-learning (Deep Q-Network)**.  

---

## 🧠 Technologies & Concepts Used

| Technology | Purpose | Why Used |
|-------------|----------|-----------|
| **Python 3** | Core programming language | Easy integration with ML libraries |
| **TensorFlow / Keras** | Deep learning model building | Supports CNN, LSTM, and Reinforcement Learning |
| **OpenCV** | Video capture and rendering | Used for saving evaluation videos |
| **NumPy / Matplotlib** | Data handling and visualization | Efficient computation & result plotting |
| **Deep Reinforcement Learning (DQN)** | Agent training | Learns optimal actions via reward feedback |
| **Experience Replay** | Memory buffer for Q-learning | Stabilizes training |
| **Double DQN + Dueling Architecture** | Enhanced learning efficiency | Improves accuracy and stability |

---

Trained Model
      │
      ▼
  Load Environment
      │
      ▼
  Predict Best Actions
      │
      ▼
  Move Agent → Collect Reward
      │
      ▼
  Save Video (Success / Failure)
      │
      ▼
  Compute Success Rate

## 📁 Project Structure

smart-vision-robot/
│
├── env/
│ └── simulation_env.py # Environment setup (Grid, rewards, agent moves)
│
├── models/
│ ├── cnn_lstm_model.py # CNN + LSTM (Dueling DQN) model architecture
│ ├── replay_buffer.py # Experience Replay memory buffer
│ └── agent.py # Agent logic: choose action, learn, update Q-network
│
├── training/
│ └── train_agent.py # Main training script for the agent
│
├── evaluation/
│ └── evaluate_agent.py # Runs trained agent & saves evaluation videos
│
├── videos/ # Folder to store output evaluation videos
│
├── requirements.txt # Python dependencies
└── summary.md # Project summary and documentation
---

## ⚙️ How It Works — Flow Explanation

### 1. Environment Setup (`simulation_env.py`)
- Creates a **grid world** where:
  - Green cell = goal (reward +10)  
  - Red cell = obstacle (penalty -5)  
  - Empty cell = free path (reward -0.1)
- The agent receives visual input (grid image) and moves (Up/Down/Left/Right).

### 2. Agent Creation (`agent.py`)
- Uses the **CNN+LSTM Dueling DQN model** from `cnn_lstm_model.py`.
- Maintains:
  - **Main Network** – Learns from experiences.
  - **Target Network** – Provides stable Q-value estimation.
- Uses **epsilon-greedy** strategy:
  - Random exploration initially.
  - Gradually shifts to exploitation (learned decisions).

### 3. Model Architecture (`cnn_lstm_model.py`)
- **CNN Layers:** Extract spatial features (like vision).
- **LSTM Layer:** Retains memory of previous frames.
- **Dueling DQN Output:** Separates *Value* and *Advantage* streams to improve stability.
- **Output Layer:** Predicts Q-values for each possible action.

### 4. Training Process (`train_agent.py`)
1. Initialize environment and networks.
2. Run for multiple **episodes** (e.g., 500–2000).  
3. For each step:
   - Get state → predict action → move → get reward → store in replay buffer.
4. Train the model by sampling from memory.
5. Update target network periodically.
6. Save model checkpoints after every few episodes.

🧩 Optimizations used:
- **Double DQN:** Prevents overestimation of Q-values.
- **Dueling Architecture:** Separates state value and action advantage.
- **Soft Updates:** Smooth synchronization between main and target networks.
- **Batch Normalization + Dropout:** Regularization for faster convergence.

---

## 🎥 Evaluation Process (`evaluate_agent.py`)
- Loads the trained model and runs multiple test episodes.
- Saves **videos** of both successful and failed runs:
  - Successful runs (agent reaches goal) → `videos/success_episode_X.mp4`
  - Failed runs → `videos/failure_episode_X.mp4`
- Each video includes:
  - Overlaid episode number and total reward.
  - Grid movement visualization.

---

## 🧩 How the Model Learns
1. **Input:** Grid image (state)
2. **Processing:**
   - CNN extracts features (like edges, goal position)
   - LSTM remembers past steps
3. **Output:** Q-values for each possible action
4. **Q-Learning Update Rule:**
   \[
   Q(s, a) = r + \gamma \cdot \max_{a'} Q'(s', a')
   \]
5. **Experience Replay:** Random batches improve stability
6. **Target Network:** Updated softly every few steps to reduce oscillation

---

| Step | Component          | Description                                     |
| ---- | ------------------ | ----------------------------------------------- |
| 1    | **Environment**    | Generates visual grid world                     |
| 2    | **CNN**            | Extracts visual spatial features                |
| 3    | **LSTM**           | Remembers past steps (temporal info)            |
| 4    | **Dueling DQN**    | Learns Q-values using state & action advantages |
| 5    | **Replay Buffer**  | Stores experience for stable training           |
| 6    | **Q-Update Loop**  | Learns from sampled experiences                 |
| 7    | **Target Network** | Provides stable learning targets                |
| 8    | **Evaluation**     | Measures success and saves videos               |

config.py      → defines hyperparameters
train_agent.py → runs full training loop
env/           → provides world + state feedback
agents/        → defines learning model and replay buffer
models/        → saves trained CNN+LSTM DQN
evaluate_*.py  → tests model and records navigation videos


| Technology               | Use                           | Why                                        |
| ------------------------ | ----------------------------- | ------------------------------------------ |
| **TensorFlow / Keras**   | Building CNN-LSTM DQN models  | Fast GPU training, simple model definition |
| **OpenCV**               | Video rendering and recording | Handles image processing and MP4 writing   |
| **NumPy**                | Array operations              | Efficient numerical computations           |
| **Gym-like Environment** | Simulated world               | Controlled training environment for RL     |
| **TensorBoard**          | Logging and visualization     | Real-time tracking of rewards and losses   |


| Phase           | Mechanism         | Explanation                                                                 |
| --------------- | ----------------- | --------------------------------------------------------------------------- |
| Perception      | CNN               | Learns spatial patterns from the environment (like visual features).        |
| Memory          | LSTM              | Keeps a short-term memory of recent frames → handles partial observability. |
| Decision Making | DQN               | Maps visual sequences to action-value pairs (Q-values).                     |
| Stability       | Target Network    | Provides stable Q-value targets to prevent divergence.                      |
| Replay          | Experience Replay | Reuses past experiences to improve sample efficiency.                       |


## 📈 Training Improvements
- Uses **Adam optimizer** with LR decay.
- Reward shaping encourages reaching the goal efficiently.
- Early stopping if high success rate achieved.

---

## ⏱️ Training Time
| Setup | Approx Time | Success Rate |
|--------|--------------|--------------|
| CPU (Intel i7) | ~6–8 hrs | 80–85% |
| GPU (RTX 3060) | ~1.5 hrs | 90–95% |

*(depends on grid size, episodes, and replay memory size)*

---
✅ Results & Insights

The agent learns to reach the goal with minimal collisions.

CNN extracts spatial features effectively.

LSTM improves decision-making by remembering paths.

Double DQN reduces instability and overfitting.

📜 Future Enhancements

Add real camera feed input.

Use PPO (Proximal Policy Optimization) for continuous control.

Extend to 3D environment (Unity MLAgents / Gazebo).


                ┌──────────────────────────────┐
                │         Environment          │
                │ (Grid world, obstacles, goal)│
                └─────────────┬────────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Visual Observation    │
                 │ (Grid image as input)     │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │       CNN Layers          │
                 │ Feature extraction from   │
                 │ environment visuals       │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │        LSTM Layer         │
                 │ Learns temporal patterns  │
                 │ (memory of past states)   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Dueling DQN Head       │
                 │ - Value Stream (V(s))    │
                 │ - Advantage Stream (A(s,a)) │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Predicted Q-values       │
                 │ for all actions          │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Action Selection         │
                 │ (ε-greedy policy)        │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Execute Action in Env     │
                 │ → New state, reward       │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │  Store Experience in      │
                 │  Replay Buffer (s, a, r, s')│
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Sample Batch & Train DQN  │
                 │ Update Q-network weights  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Soft Update Target Net    │
                 │ for stable learning       │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Next Episode Begins     │
                 │   Repeat until convergence│
                 └──────────────────────────┘
