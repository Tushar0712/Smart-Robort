# agents/dqn_agent.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os

class LSTM_DQN_Agent:
    def __init__(self, input_shape=(4,7,7,3), n_actions=5, lr=1e-4, gamma=0.99, lstm_units=64,
                 soft_update=True, tau=0.01, grad_clip=10.0):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.lstm_units = lstm_units
        self.soft_update = soft_update
        self.tau = tau
        self.grad_clip = grad_clip

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network(hard=True)

        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.Huber()

    def build_model(self):
        inp = layers.Input(shape=self.input_shape) 
        x = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu'))(inp)
        x = layers.TimeDistributed(layers.Conv2D(64, 3, activation='relu'))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(self.lstm_units)(x)
        x = layers.Dense(128, activation='relu')(x)
        qvals = layers.Dense(self.n_actions, activation=None)(x)
        return models.Model(inp, qvals)

    def update_target_network(self, hard=False):
        if hard or not self.soft_update:
            self.target_model.set_weights(self.model.get_weights())
        else:
            online_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            new_weights = [self.tau * ow + (1 - self.tau) * tw for ow, tw in zip(online_weights, target_weights)]
            self.target_model.set_weights(new_weights)

    def act(self, state_seq, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        s = np.array(state_seq)
        if s.ndim == 3:
            s = np.stack([s] * self.input_shape[0], axis=0)
        s = np.expand_dims(s, axis=0).astype(np.float32)
        qvals = self.model.predict(s, verbose=0)
        return int(np.argmax(qvals[0]))

    def train_on_batch(self, obs_seq, actions, rewards, next_obs_seq, dones):
        """
        obs_seq: (batch, time, H, W, C)
        actions: (batch,)
        rewards: (batch,)
        next_obs_seq: (batch, time, H, W, C)
        dones: (batch,)
        """
        obs_seq = obs_seq.astype(np.float32)
        next_obs_seq = next_obs_seq.astype(np.float32)
        actions = actions.astype(np.int32)
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)

        with tf.GradientTape() as tape:
            q_vals = self.model(obs_seq, training=True)  # (batch, n_actions)
            q_action = tf.reduce_sum(q_vals * tf.one_hot(actions, self.n_actions), axis=1)
            q_next = self.target_model(next_obs_seq, training=False)
            q_next_max = tf.reduce_max(q_next, axis=1)
            targets = rewards + (1.0 - dones) * self.gamma * q_next_max
            loss = self.loss_fn(targets, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        # gradient clipping
        grads = [None if g is None else tf.clip_by_norm(g, self.grad_clip) for g in grads]
        self.optimizer.apply_gradients([(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None])
        return float(loss.numpy())

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        p = os.path.join(path, "lstm_dqn_model.keras")
        self.model.save(p)
        print("Model saved to:", p)

    def load(self, path):
        p = os.path.join(path, "lstm_dqn_model.keras")
        self.model = tf.keras.models.load_model(p)
        # rebuild target model to same architecture then copy weights
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        print("Loaded model from:", p)
