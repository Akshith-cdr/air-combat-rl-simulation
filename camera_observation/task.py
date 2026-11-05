"""Camera measurement scheduling only; no aircraft or weapon actions."""

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    decision_dt: float = 0.2
    horizon: int = 240
    capture_budget: int = 24
    freshness_seconds: float = 2.0
    horizontal_fov_deg: float = 80.0
    vertical_fov_deg: float = 55.0
    capture_cost: float = 0.12
    unusable_capture_cost: float = 0.25
    fresh_observation_reward: float = 0.08
    missed_observation_cost: float = 0.03

    def __post_init__(self):
        if self.horizon < 1 or self.capture_budget < 1:
            raise ValueError('Positive horizon and capture budget required')
        if self.decision_dt <= 0 or self.freshness_seconds <= 0:
            raise ValueError('Positive camera timing required')
        for fov in (self.horizontal_fov_deg, self.vertical_fov_deg):
            if not 0 < fov < 180:
                raise ValueError('Camera FOV must be between 0 and 180 degrees')


def framing(frame):
    """Symbolic forward camera: u/v are normalized image-plane coordinates.

    Visibility comes from geometric projection, not an image detector or radar.
    A frame contains u, v and a positive-depth flag; no range/weapon features.
    """
    u, v, forward = map(float, frame[:3])
    visible = bool(forward > 0 and abs(u) <= 1 and abs(v) <= 1)
    quality = max(0.0, 1.0 - (u*u + v*v) / 2) if visible else 0.0
    return u, v, visible, quality


class CameraState:
    def __init__(self, config=CameraConfig()):
        self.config = config
        self.step_index = 0
        self.remaining = config.capture_budget
        self.age = math.inf
        self.last_quality = 0.0
        self.previous_uv = None
        self.rows = []

    def observation(self, frame):
        u, v, visible, quality = framing(frame)
        previous = self.previous_uv if self.previous_uv is not None else (u, v)
        result = [u/2, v/2, (u-previous[0])/2, (v-previous[1])/2,
                  float(visible), quality, min(1.0, self.age/self.config.freshness_seconds),
                  self.remaining/self.config.capture_budget,
                  self.step_index/self.config.horizon, self.last_quality]
        return np.clip(result, -1, 1).astype(np.float32)

    def record(self, action, frame):
        if action not in (0, 1):
            raise ValueError('Camera action must be 0 (wait) or 1 (record sighting)')
        u, v, visible, quality = framing(frame)
        available = self.remaining > 0
        recorded = bool(action == 1 and visible and available)
        reward = 0.0
        if action == 1:
            if available:
                self.remaining -= 1
            reward -= self.config.capture_cost if recorded else self.config.unusable_capture_cost
        if recorded:
            self.age, self.last_quality = 0.0, quality
        fresh = self.age < self.config.freshness_seconds
        if visible:
            reward += (self.config.fresh_observation_reward * self.last_quality
                       * max(0.0, 1-self.age/self.config.freshness_seconds)
                       if fresh else -self.config.missed_observation_cost)
        row = dict(step=self.step_index, time_s=self.step_index*self.config.decision_dt,
                   action=int(action), visible=visible, quality=quality, recorded=recorded,
                   fresh=bool(fresh), remaining=self.remaining, reward=float(reward))
        self.rows.append(row)
        self.previous_uv = (u, v)
        self.age += self.config.decision_dt
        self.step_index += 1
        return float(reward), row

    def metrics(self, reason='horizon'):
        visible = [r for r in self.rows if r['visible']]
        captured = [r for r in self.rows if r['recorded']]
        requests = [r for r in self.rows if r['action']]
        return dict(acquired=bool(captured), captures=len(captured), requests=len(requests),
                    unusable_requests=len(requests)-len(captured),
                    first_acquisition_s=captured[0]['time_s'] if captured else None,
                    visible_steps=len(visible), steps=len(self.rows),
                    visible_tracking_coverage=(sum(r['fresh'] for r in visible)/len(visible)
                                               if visible else None),
                    overall_track_availability=(sum(r['fresh'] for r in self.rows)/len(self.rows)
                                                if self.rows else 0.0),
                    mean_capture_quality=(float(np.mean([r['quality'] for r in captured]))
                                          if captured else None),
                    episode_return=sum(r['reward'] for r in self.rows),
                    termination_reason=reason)


class CameraObservationEnv(gym.Env):
    """Replay exact frozen flights; actions change camera bookkeeping only.

    Flights are recorded using original FighterJetEnv physics. Replay is exact
    because recording a sighting cannot influence either aircraft's motion.
    """
    def __init__(self, traces, config=CameraConfig()):
        self.traces, self.config = traces, config
        if not traces:
            raise ValueError('At least one original-simulator flight is required')
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-1, 1, (10,), dtype=np.float32)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get('trace_index', int(self.np_random.integers(len(self.traces))))
        self.trace = self.traces[index]
        self.state = CameraState(self.config)
        return self.state.observation(self.trace['frames'][0]), {'trace_seed': self.trace['seed']}

    def step(self, action):
        if self.state.step_index >= len(self.trace['frames'])-1:
            raise RuntimeError('Episode finished; reset before stepping')
        reward, info = self.state.record(int(action), self.trace['frames'][self.state.step_index])
        done = self.state.step_index >= len(self.trace['frames'])-1
        reason = self.trace['termination_reason']
        terminated, truncated = done and reason != 'horizon', done and reason == 'horizon'
        observation = self.state.observation(self.trace['frames'][self.state.step_index])
        if done:
            info = {**info, 'camera_metrics': self.state.metrics(reason)}
        return observation, reward, terminated, truncated, info
