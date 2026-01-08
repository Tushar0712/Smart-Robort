# env/simulation_env.py
import numpy as np
import cv2

class GridEnvironment:
    """
    Small grid environment returning local RGB patch as observation.
    """
    def __init__(self, height=12, width=12, n_obstacles=15, vision_size=7, max_steps=200, seed=None):
        self.H = height
        self.W = width
        self.n_obstacles = n_obstacles
        self.vision_size = vision_size
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)
        # place obstacles
        for _ in range(self.n_obstacles):
            x, y = self.rng.randint(0, self.H), self.rng.randint(0, self.W)
            self.grid[x, y] = 1
        # agent
        while True:
            ax, ay = self.rng.randint(0, self.H), self.rng.randint(0, self.W)
            if self.grid[ax, ay] == 0:
                self.agent_pos = (ax, ay)
                break
        # goal
        while True:
            gx, gy = self.rng.randint(0, self.H), self.rng.randint(0, self.W)
            if self.grid[gx, gy] == 0 and (gx, gy) != self.agent_pos:
                self.goal_pos = (gx, gy)
                break
        self.steps = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        pad = self.vision_size // 2
        padded = np.pad(self.grid, pad, mode='constant', constant_values=1)
        ax, ay = self.agent_pos
        axp, ayp = ax + pad, ay + pad
        patch = padded[axp-pad:axp+pad+1, ayp-pad:ayp+pad+1]
        h, w = patch.shape
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        img[patch == 1] = (30, 30, 30)  
        gx, gy = self.goal_pos
        rel_gx, rel_gy = gx - (ax - pad), gy - (ay - pad)
        if 0 <= rel_gx < h and 0 <= rel_gy < w:
            img[rel_gx, rel_gy] = (50, 200, 50)  # goal color
        center = pad
        img[center, center] = (200, 50, 50)  # agent center
        return img.astype(np.float32) / 255.0

    def step(self, action):
        if self.done:
            return self.reset(), 0.0, True, {}
        ax, ay = self.agent_pos
        if action == 0:
            nx, ny = ax - 1, ay
        elif action == 1:
            nx, ny = ax + 1, ay
        elif action == 2:
            nx, ny = ax, ay - 1
        elif action == 3:
            nx, ny = ax, ay + 1
        else:
            nx, ny = ax, ay
        nx = max(0, min(self.H - 1, nx))
        ny = max(0, min(self.W - 1, ny))
        reward = -0.01
        if self.grid[nx, ny] == 1:
            reward -= 0.5
            nx, ny = ax, ay  # collision -> stay
        else:
            self.agent_pos = (nx, ny)
        # optional shaping: small reward for getting closer
        old_dist = abs(ax - self.goal_pos[0]) + abs(ay - self.goal_pos[1])
        new_dist = abs(nx - self.goal_pos[0]) + abs(ny - self.goal_pos[1])
        reward += 0.01 * (old_dist - new_dist)
        if (nx, ny) == self.goal_pos:
            reward += 1.0
            self.done = True
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        return self._get_observation(), float(reward), bool(self.done), {}

    def _render_grid_image(self):
        cell_size = 20
        img = np.ones((self.H * cell_size, self.W * cell_size, 3), dtype=np.uint8) * 255
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i, j] == 1:
                    img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = (50, 50, 50)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        img[ax*cell_size:(ax+1)*cell_size, ay*cell_size:(ay+1)*cell_size] = (200, 50, 50)
        img[gx*cell_size:(gx+1)*cell_size, gy*cell_size:(gy+1)*cell_size] = (50, 200, 50)
        for x in range(self.H + 1):
            cv2.line(img, (0, x*cell_size), (self.W*cell_size, x*cell_size), (200, 200, 200), 1)
        for y in range(self.W + 1):
            cv2.line(img, (y*cell_size, 0), (y*cell_size, self.H*cell_size), (200, 200, 200), 1)
        return img

    def render(self, scale=2):
        img = self._render_grid_image()
        resized = cv2.resize(img, (self.W*20*scale, self.H*20*scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Grid", resized)
        cv2.waitKey(1)
