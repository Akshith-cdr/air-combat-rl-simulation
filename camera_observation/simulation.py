"""Read-only frozen flight outputs and the original JetSim physics backend."""

import hashlib
import math
from pathlib import Path
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO
from panda3d.core import Vec3

from fighter_jet_env import FighterJetEnv, DEFAULT_SIMULATION_DT
from .task import CameraConfig


ROOT = Path(__file__).resolve().parents[1]
FLIGHT_MODEL = ROOT/'experiments/flight_policy/flight_v2_seed63011_run03/final/candidate.zip'
FLIGHT_STATS = FLIGHT_MODEL.with_name('candidate_vecnormalize.pkl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenFlightController:
    def __init__(self, model_path=FLIGHT_MODEL, stats_path=FLIGHT_STATS):
        self.paths = [Path(model_path).resolve(), Path(stats_path).resolve()]
        self.hashes = {str(p): digest(p) for p in self.paths}
        self.model = PPO.load(str(model_path), device='cpu')
        self.model.policy.set_training_mode(False)
        self.model.policy.requires_grad_(False)
        with Path(stats_path).open('rb') as stream:
            self.normalizer = pickle.load(stream)
        self.normalizer.training = self.normalizer.norm_reward = False
        self.moments = (self.normalizer.obs_rms.mean.copy(), self.normalizer.obs_rms.var.copy(),
                        float(self.normalizer.obs_rms.count))

    def controls(self, observation):
        normalized = self.normalizer.normalize_obs(np.asarray(observation, dtype=np.float32)[None])
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        with torch.no_grad():
            policy = self.model.policy
            features = policy.extract_features(tensor)
            latent = policy.mlp_extractor.forward_actor(features)
            # Read exactly the frozen four flight means. Never query a fire head
            # or the old classifier; camera actions have no route into that lane.
            flight = policy.action_net(latent).cpu().numpy()[0]
        return np.clip(flight, -1, 1).astype(np.float32)

    def verify(self):
        if self.hashes != {str(p): digest(p) for p in self.paths}:
            raise RuntimeError('Frozen flight checkpoint or statistics changed')
        for actual, expected in zip((self.normalizer.obs_rms.mean, self.normalizer.obs_rms.var,
                                     self.normalizer.obs_rms.count), self.moments):
            np.testing.assert_array_equal(actual, expected)


def scenario(seed):
    rng = np.random.default_rng(seed)
    distance, bearing = rng.uniform(1800, 4500), math.radians(rng.uniform(-75, 75))
    altitude = float(rng.uniform(750, 1500))
    return dict(scenario_id=f'CAMERA-{seed}', seed=int(seed),
                enemy_initial_position=(float(math.sin(bearing)*distance),
                                        float(math.cos(bearing)*distance), altitude),
                enemy_initial_altitude=altitude, enemy_initial_speed=float(rng.uniform(140, 220)),
                enemy_initial_heading=float(rng.uniform(0, 360)))


class OriginalFlightSession:
    def __init__(self, controller, config=CameraConfig(), headless=True):
        self.controller, self.config = controller, config
        self.stride = round(config.decision_dt/DEFAULT_SIMULATION_DT)
        if self.stride < 1 or not math.isclose(self.stride*DEFAULT_SIMULATION_DT, config.decision_dt):
            raise ValueError('Decision period must be an integer number of original physics steps')
        self.env = FighterJetEnv(hybrid_fire=True, headless=headless)
        # Instance-local fail-closed tripwires. Original weapon implementations
        # and all other simulation modes are untouched.
        def forbidden(*args, **kwargs):
            raise RuntimeError('Camera observation task cannot launch a weapon')
        self.env.sim.fire_missile = forbidden
        self.env.sim.launch_guided_missile = forbidden

    def reset(self, seed):
        self.seed = int(seed)
        self.observation, _ = self.env.reset(seed=seed, options={'enemy_challenge_scenario': scenario(seed)})
        self.reason, self.decisions = 'horizon', 0
        return self.frame()

    def frame(self):
        sim = self.env.sim
        relative = sim.player.root.getQuat().conjugate().xform(sim.enemy.position-sim.player.position)
        depth = float(relative.y)
        # Clipping preserves the camera's out-of-frame state without infinities.
        denominator = max(abs(depth), 1e-6)
        u = float(relative.x)/(denominator*math.tan(math.radians(self.config.horizontal_fov_deg/2)))
        v = float(relative.z)/(denominator*math.tan(math.radians(self.config.vertical_fov_deg/2)))
        return np.array([np.clip(u, -4, 4), np.clip(v, -4, 4), float(depth > 0)], dtype=np.float64)

    def advance(self, before_step=None, after_step=None):
        for _ in range(self.stride):
            if before_step:
                before_step(self.env.sim)
            controls = self.controller.controls(self.observation)
            self.last_controls = controls.copy()
            self.observation, _, done, truncated, info = self.env.step(np.r_[controls, np.float32(0)])
            if self.env.sim.active_missiles or info['weapon']['fire_request']:
                raise RuntimeError('Camera task violated its non-weapon contract')
            if after_step:
                after_step(self.env.sim)
            if done or truncated:
                self.reason = 'flight_boundary' if 'BOUNDARY' in info['terminal_reason'] else 'flight_terminated'
                break
        self.decisions += 1
        return self.frame()

    def close(self):
        self.env.close()


def collect_trace(controller, seed, config=CameraConfig()):
    session = OriginalFlightSession(controller, config)
    try:
        frames = [session.reset(seed)]
        positions = [tuple(session.env.sim.player.position)]
        controls = []
        for _ in range(config.horizon):
            frames.append(session.advance())
            positions.append(tuple(session.env.sim.player.position))
            controls.append(tuple(session.last_controls))
            if session.reason != 'horizon':
                break
        position_array = np.asarray(positions)
        return dict(seed=int(seed), frames=np.asarray(frames), positions=position_array,
                    controls=np.asarray(controls), termination_reason=session.reason,
                    flight_path_m=float(np.linalg.norm(np.diff(position_array, axis=0), axis=1).sum()),
                    minimum_sampled_altitude_m=float(position_array[:, 2].min()))
    finally:
        session.close()
