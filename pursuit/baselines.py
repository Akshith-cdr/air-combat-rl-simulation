"""Hand-written pursuit controllers used as baselines (reviewer comment M3).

Both use the same four controls as the PPO policy and fly in the same
PursuitEnv, under the same launch rule, gate, missile and episodes.

pure pursuit  aim the velocity vector at the target's current position.
lead pursuit  aim it at the point the target will reach after the predicted
              time to go, t_go = range / closing speed (clipped to 0..max_lead_s),
              while outside the launch band; inside the band, aim at the
              target itself, because the launch gate requires the velocity to
              point at the target (the common lead-then-pure scheme).

Speed: hold the target's speed plus a range term, so the aircraft closes when
far and matches the target's speed at `hold_range_m`.
Read from the simulator state directly (perfect sensing, like the policy).

The gains of each controller were chosen by a grid search on a separate tuning
suite (pursuit.tune_baselines) and are stored in baseline_gains.json.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pursuit import config as C

GAINS_FILE = Path(__file__).with_name("baseline_gains.json")


@dataclass(frozen=True)
class Gains:
    bank_per_deg: float = 1.5        # commanded bank per degree of heading error
    yaw_deg_per_unit: float = 20.0   # heading error (deg) that gives a full yaw command
    hold_range_m: float = 750.0      # range at which the target's speed is matched
    speed_per_m: float = 0.1         # extra commanded speed per metre beyond hold_range_m
    max_bank_deg: float = 75.0
    max_lead_s: float = 5.0
    min_speed: float = 160.0
    max_extra_speed: float = 200.0


def _heading(v):
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


def _elevation(v):
    return math.degrees(math.atan2(v[2], max(1e-8, math.hypot(v[0], v[1]))))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class PursuitController:
    def __init__(self, lead=False, gains=None):
        self.lead = bool(lead)
        self.name = "lead_pursuit" if lead else "pure_pursuit"
        self.gains = gains or Gains()

    def reset(self):
        pass

    def aim_point(self, p, vp, t, vt):
        rel = t - p
        d = float(np.linalg.norm(rel))
        if not self.lead or d <= C.BAND_MAX_M:
            return t
        closing = -float((vt - vp) @ rel) / max(d, 1e-6)
        t_go = float(np.clip(d / max(closing, 1.0), 0.0, self.gains.max_lead_s))
        return t + vt * t_go

    def act(self, env, obs=None):
        g = self.gains
        player, enemy = env.sim.player, env.sim.enemy
        p, vp = np.array(player.position, float), np.array(player.velocity, float)
        t, vt = np.array(enemy.position, float), np.array(enemy.velocity, float)
        aim = self.aim_point(p, vp, t, vt) - p

        heading_error = _wrap(_heading(aim) - _heading(vp))        # + = aim is to the right
        desired_pitch = float(np.clip(_elevation(aim), -45.0, 45.0))

        desired_bank = float(np.clip(g.bank_per_deg * heading_error, -g.max_bank_deg, g.max_bank_deg))
        roll = float(np.clip((desired_bank - player.roll) / 20.0, -1, 1))          # + roll = right turn
        yaw = float(np.clip(-heading_error / g.yaw_deg_per_unit, -1, 1))           # + yaw = left turn
        pitch = float(np.clip((desired_pitch - player.pitch) / 15.0, -1, 1))

        distance = float(np.linalg.norm(t - p))
        wanted = float(np.linalg.norm(vt)) + float(np.clip(
            g.speed_per_m * (distance - g.hold_range_m), -g.max_extra_speed, g.max_extra_speed))
        wanted = max(g.min_speed, wanted)
        error = wanted - player.target_speed
        throttle = 0.0 if abs(error) < 2.0 else float(np.sign(error))
        return np.array([pitch, roll, yaw, throttle], dtype=np.float32)


def load_gains(name):
    """Tuned gains for a baseline, or the defaults if tuning has not been run."""
    if GAINS_FILE.exists():
        table = json.loads(GAINS_FILE.read_text())
        if name in table:
            return Gains(**table[name]["gains"])
    return Gains()


POLICY_FILE = "policy_numpy.npz"


def export_policy(directory):
    """Write the deterministic actor of model.zip + vecnormalize.pkl as plain arrays.

    Evaluation workers then need only numpy: on Windows every process that
    loads torch reserves about 1 GB, which limits how many can run at once.
    """
    import pickle
    from pathlib import Path
    from stable_baselines3 import PPO
    directory = Path(directory)
    model = PPO.load(str(directory / "model.zip"), device="cpu")
    with open(directory / "vecnormalize.pkl", "rb") as handle:
        stats = pickle.load(handle)
    layers = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    activation = type(model.policy.mlp_extractor.policy_net[1]).__name__
    if activation != "Tanh":
        raise ValueError(f"expected Tanh hidden layers, found {activation}")
    arrays = {f"w{i}": layer.weight.detach().numpy() for i, layer in enumerate(layers)}
    arrays.update({f"b{i}": layer.bias.detach().numpy() for i, layer in enumerate(layers)})
    arrays.update(w_out=model.policy.action_net.weight.detach().numpy(),
                  b_out=model.policy.action_net.bias.detach().numpy(),
                  obs_mean=stats.obs_rms.mean, obs_var=stats.obs_rms.var,
                  clip_obs=np.array(stats.clip_obs), epsilon=np.array(stats.epsilon),
                  low=model.action_space.low, high=model.action_space.high)
    np.savez(directory / POLICY_FILE, **arrays)


class PPOController:
    """Deterministic (mean) action of a saved PPO actor, evaluated with numpy."""

    def __init__(self, directory):
        from pathlib import Path
        p = np.load(Path(directory) / POLICY_FILE)
        self.hidden = [(p[f"w{i}"], p[f"b{i}"]) for i in range(sum(k.startswith("w") and k[1:].isdigit()
                                                                   for k in p.files))]
        self.w_out, self.b_out = p["w_out"], p["b_out"]
        self.mean, self.var = p["obs_mean"], p["obs_var"]
        self.clip, self.eps = float(p["clip_obs"]), float(p["epsilon"])
        self.low, self.high = p["low"], p["high"]
        self.name = "ppo"

    def reset(self):
        pass

    def act(self, env, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)
        for w, b in self.hidden:
            x = np.tanh(w @ x + b)
        return np.clip(self.w_out @ x + self.b_out, self.low, self.high)


def make(name, gains=None):
    """A baseline by name, with explicit gains or the tuned ones."""
    if name not in ("pure_pursuit", "lead_pursuit"):
        raise KeyError(name)
    return PursuitController(lead=name == "lead_pursuit", gains=gains or load_gains(name))
