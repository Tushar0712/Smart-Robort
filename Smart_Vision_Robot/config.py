# config.py
class Config:
    SEED = 42
    TIME_STEPS = 4
    BATCH_SIZE = 32
    BUFFER_CAPACITY = 100000
    MIN_REPLAY_SIZE = 2000
    EPISODES = 1000
    LEARNING_STARTS = 2000
    TARGET_UPDATE_EVERY = 50        
    SOFT_UPDATE = True             
    TAU = 0.01                      
    LR = 1e-4
    GAMMA = 0.99
    LSTM_UNITS = 64
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    ENV = dict(height=12, width=12, n_obstacles=20, vision_size=7, max_steps=100)
    SAVE_DIR = "models"
    LOG_DIR = "logs"
