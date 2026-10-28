"""Shared live-only wrapper: retain the terminal scene for impact animation."""

import gymnasium as gym
import numpy as np


class PreserveTerminalVisuals(gym.Wrapper):
    """Prevent DummyVecEnv's auto-reset from erasing the confirmed hit scene."""

    def __init__(self, env):
        super().__init__(env)
        self._terminal_observation = None

    def reset(self, *, seed=None, options=None):
        if self._terminal_observation is not None and seed is None and options is None:
            observation = self._terminal_observation
            self._terminal_observation = None
            return observation.copy(), {"terminal_visuals_preserved": True}
        self._terminal_observation = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        result = self.env.step(action)
        observation, _, terminated, truncated, _ = result
        if terminated or truncated:
            self._terminal_observation = np.asarray(observation, dtype=np.float32).copy()
        return result
