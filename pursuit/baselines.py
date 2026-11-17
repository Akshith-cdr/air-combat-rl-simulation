"""Hand-written controllers used as baselines (reviewer comments M1, M3, M4).

All use the same four controls as the PPO policy and fly in the same
PursuitEnv, under the same launch rule, gate, missile and episodes.

pure pursuit   aim the velocity vector at the target's current position.
lead pursuit   aim it at the point the target will reach after the predicted
               time to go, t_go = range / closing speed (clipped to 0..max_lead_s),
               while outside the launch band; inside the band, point at the
               target itself, because the launch gate requires the velocity to
               point at the target ("lead-then-point").
PN pursuit     proportional navigation of the aircraft itself: command a heading
               (and flight-path) change of nav_gain x the line-of-sight rate x 1 s,
               which drives the line-of-sight rate to zero (a collision course).
               PN cannot start a turn towards a target far off the nose (the
               line of sight barely rotates), so beyond pn_max_ata_deg it uses
               pure pursuit; inside the band it points at the target
               ("PN-then-point").

The bank/yaw/pitch inner loops and the speed rule are shared. Speed rule: hold
the target's speed plus a range term, so the aircraft closes when far and
matches the target's speed at `hold_range_m`.

Speed-swap hybrids (M4): SpeedSwap takes the attitude commands (pitch, roll,
yaw) from one controller and the throttle from another: PPO steering with the
baseline speed rule, or a baseline's steering with PPO's throttle.

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
    nav_gain: float = 3.0            # PN: commanded turn = nav_gain x LOS rate x 1 s
    pn_max_ata_deg: float = 60.0     # PN: pure pursuit beyond this angle off the nose
    min_speed: float = 160.0
    max_extra_speed: float = 200.0


def _heading(v):
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


def _elevation(v):
    return math.degrees(math.atan2(v[2], max(1e-8, math.hypot(v[0], v[1]))))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class PursuitController:
    KINDS = {"pure_pursuit": "pure", "lead_pursuit": "lead", "pn_pursuit": "pn"}

    def __init__(self, kind="pure", gains=None):
        if kind not in ("pure", "lead", "pn"):
            raise ValueError(kind)
        self.kind = kind
        self.lead = kind == "lead"
        self.name = {v: k for k, v in self.KINDS.items()}[kind]
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

    def pn_command(self, p, vp, t, vt, pitch):
        """(heading error, desired pitch) in degrees from proportional navigation,
        or None when pure pursuit applies (inside the band or target far off the nose).

        The vertical correction is added to the nose pitch, not to the velocity's
        flight-path angle: in this simulator the velocity sinks below the nose
        (e.g. -63 deg flight path at -44 deg pitch), and referencing it makes the
        pitch loop dive away."""
        rel, vrel = t - p, vt - vp
        d = float(np.linalg.norm(rel))
        cos_ata = float(rel @ vp) / max(d * float(np.linalg.norm(vp)), 1e-9)
        ata = math.degrees(math.acos(float(np.clip(cos_ata, -1.0, 1.0))))
        if d <= C.BAND_MAX_M or ata > self.gains.pn_max_ata_deg:
            return None
        x, y, z = rel
        h2 = max(x * x + y * y, 1e-6)
        h = math.sqrt(h2)
        az_rate = (y * vrel[0] - x * vrel[1]) / h2                    # rad/s, + = clockwise (right)
        h_dot = (x * vrel[0] + y * vrel[1]) / h
        el_rate = (h * vrel[2] - z * h_dot) / (h2 + z * z)             # rad/s, + = up
        n = self.gains.nav_gain
        heading_error = float(np.clip(math.degrees(n * az_rate), -90.0, 90.0))
        desired_pitch = float(np.clip(pitch + math.degrees(n * el_rate), -45.0, 45.0))
        return heading_error, desired_pitch

    def steer(self, env):
        """Pitch, roll and yaw commands."""
        g = self.gains
        player, enemy = env.sim.player, env.sim.enemy
        p, vp = np.array(player.position, float), np.array(player.velocity, float)
        t, vt = np.array(enemy.position, float), np.array(enemy.velocity, float)
        command = self.pn_command(p, vp, t, vt, player.pitch) if self.kind == "pn" else None
        if command is None:
            aim = self.aim_point(p, vp, t, vt) - p
            heading_error = _wrap(_heading(aim) - _heading(vp))    # + = aim is to the right
            desired_pitch = float(np.clip(_elevation(aim), -45.0, 45.0))
        else:
            heading_error, desired_pitch = command

        desired_bank = float(np.clip(g.bank_per_deg * heading_error, -g.max_bank_deg, g.max_bank_deg))
        roll = float(np.clip((desired_bank - player.roll) / 20.0, -1, 1))          # + roll = right turn
        yaw = float(np.clip(-heading_error / g.yaw_deg_per_unit, -1, 1))           # + yaw = left turn
        pitch = float(np.clip((desired_pitch - player.pitch) / 15.0, -1, 1))
        return np.array([pitch, roll, yaw], dtype=np.float32)

    def throttle(self, env):
        """The speed rule: target speed + range term, bang-bang on the commanded speed."""
        g = self.gains
        player, enemy = env.sim.player, env.sim.enemy
        distance = float(np.linalg.norm(np.array(enemy.position, float) - np.array(player.position, float)))
        wanted = float(np.linalg.norm(np.array(enemy.velocity, float))) + float(np.clip(
            g.speed_per_m * (distance - g.hold_range_m), -g.max_extra_speed, g.max_extra_speed))
        wanted = max(g.min_speed, wanted)
        error = wanted - player.target_speed
        return 0.0 if abs(error) < 2.0 else float(np.sign(error))

    def act(self, env, obs=None):
        return np.append(self.steer(env), np.float32(self.throttle(env))).astype(np.float32)


class SpeedSwap:
    """Attitude commands from `steer`, throttle from `speed` (reviewer comment M4).

    Both controllers see the same state on every step; only their outputs are mixed.
    """

    def __init__(self, steer, speed, name):
        self.steer_ctrl, self.speed_ctrl, self.name = steer, speed, name

    def reset(self):
        self.steer_ctrl.reset()
        self.speed_ctrl.reset()

    def act(self, env, obs):
        action = np.array(self.steer_ctrl.act(env, obs), dtype=np.float32)
        action[3] = np.float32(self.speed_ctrl.act(env, obs)[3])
        return action


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


BASELINES = tuple(PursuitController.KINDS)
SWAPS = ("ppo_rule_speed", "rule_steer_ppo_speed")


def best_baseline():
    """The baseline with the most hits on the tuning suite (ties: first in BASELINES).

    The speed-swap experiments use this one; it is chosen before any test episode is flown."""
    if not GAINS_FILE.exists():
        raise FileNotFoundError(f"{GAINS_FILE} is missing: run pursuit.tune_baselines first")
    table = json.loads(GAINS_FILE.read_text())
    return max(BASELINES, key=lambda b: (table[b]["tuning_hits"], -BASELINES.index(b)))


def make(name, gains=None):
    """A baseline by name, with explicit gains or the tuned ones."""
    if name not in PursuitController.KINDS:
        raise KeyError(name)
    return PursuitController(PursuitController.KINDS[name], gains=gains or load_gains(name))


def make_swap(kind, model, baseline=None):
    """'ppo_rule_speed': PPO steering + the baseline's speed rule.
    'rule_steer_ppo_speed': the baseline's steering + PPO's throttle."""
    baseline = baseline or best_baseline()
    rule, ppo = make(baseline), PPOController(model)
    if kind == "ppo_rule_speed":
        return SpeedSwap(ppo, rule, f"ppo_steer+{baseline}_speed")
    if kind == "rule_steer_ppo_speed":
        return SpeedSwap(rule, ppo, f"{baseline}_steer+ppo_speed")
    raise KeyError(kind)
