"""PursuitEnv: learned pursuit flight, fixed launch rule, fixed PN missile.

Action (4): pitch, roll, yaw and throttle commands in [-1, 1].
Observation (19): the same relative-geometry vector as FighterJetEnv.

Firing is not learned. Until the single missile has left the rail, a launch is
requested on every step; the gate accepts it only when
BAND_MIN_M <= range <= BAND_MAX_M and cos(ATA) > COS_ATA_GATE.
So the missile launches on the first step the aircraft is inside the band
and pointing at the target. The same rule is used in training and evaluation.

An episode ends with exactly one of these outcomes:
  hit          missile proximity hit (success)
  collision    our aircraft came within COLLISION_RADIUS_M of the target
  ground       our aircraft went below z = 0 (there is no terrain otherwise)
  missile_*    the launched missile was removed without a hit (expired, floor, range)
  boundary     the aircraft left the 10 km chase boundary
  timeout      MAX_EPISODE_STEPS reached (gymnasium truncation)
"""

import math

import gymnasium as gym
import numpy as np
from panda3d.core import Vec3

from fighter_jet_env import FighterJetEnv
from pursuit import config as C
from pursuit.scenarios import Scenario, training_scenario

SUCCESS = "hit"


def band_excess(distance):
    """Metres outside the launch band (0 inside it)."""
    return max(0.0, distance - C.BAND_MAX_M) + max(0.0, C.BAND_MIN_M - distance)


def segment_min_distance(rel_start, rel_end):
    """Closest approach of a relative position that moves linearly over one step."""
    a, b = np.asarray(rel_start, float), np.asarray(rel_end, float)
    d = b - a
    denom = float(d @ d)
    t = 0.0 if denom <= 1e-12 else float(np.clip(-(a @ d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + t * d))


def launch_geometry(rel, velocities):
    """Target aspect and closing speed at the moment of launch (diagnostics only).

    aspect_deg: angle between the target's velocity and the line from the target
    back to us. 0 = target flying straight at us (head-on), 180 = we are
    directly behind it (tail chase).
    """
    vp, vt = velocities
    rel = np.asarray(rel, float)
    d = float(np.linalg.norm(rel))
    to_shooter = -rel / max(d, 1e-9)
    speed = float(np.linalg.norm(vt))
    aspect = math.degrees(math.acos(float(np.clip(vt @ to_shooter / max(speed, 1e-9), -1, 1))))
    closing = -float((vt - vp) @ (rel / max(d, 1e-9)))
    return {"aspect_deg": aspect, "closing_speed_mps": closing}


def reward_terms(prev_d, d, prev_cos, cos_ata, speed, altitude, aoa, launched_before):
    """Per-step reward (Eq. 2 of the paper); event rewards are added by the caller.

    Before launch the engagement terms pull the aircraft into the launch band
    and onto the line of sight. After launch they are zero, so nothing rewards
    flying closer to the target while the missile is in the air.
    """
    terms = dict(closure=0.0, alignment=0.0, geometry=0.0)
    if not launched_before:
        dt = C.SIMULATION_DT
        closure = float(np.clip((band_excess(prev_d) - band_excess(d)) / (C.CLOSURE_SPEED_SCALE * dt), -1, 1))
        aligned = float(np.clip((cos_ata - C.COS_ATA_GATE) / (1 - C.COS_ATA_GATE), 0, 1))

        def potential(dist, cos_value):
            near = float(np.clip(1 - band_excess(dist) / C.GEOMETRY_RANGE_SCALE, 0, 1))
            return near * (cos_value + 1) / 2

        geometry = float(np.clip((potential(d, cos_ata) - potential(prev_d, prev_cos))
                                 / C.GEOMETRY_PROGRESS_SCALE, -1, 1))
        terms.update(closure=closure, alignment=max(0.0, closure) * aligned, geometry=geometry)
    terms["speed"] = -float(np.clip((C.MIN_SAFE_SPEED - speed) / (0.5 * C.MIN_SAFE_SPEED), 0, 1))
    terms["altitude"] = -float(np.clip((C.MIN_SAFE_ALTITUDE - altitude) / (0.5 * C.MIN_SAFE_ALTITUDE), 0, 1))
    terms["aoa"] = -float(np.clip((abs(aoa) - C.MAX_SAFE_AOA_DEG) / 30.0, 0, 1))
    return terms


class PursuitEnv(FighterJetEnv):
    def __init__(self, train_seed=0, record_trajectory=False):
        self._train_rng = np.random.default_rng(train_seed)
        self._train_episode = 0
        self.record_trajectory = bool(record_trajectory)
        self.scenario = None
        self.trajectory = []
        super().__init__(hybrid_fire=True, missile_guidance=True, headless=True,
                         simulation_dt=C.SIMULATION_DT, launch_range_gate=C.BAND_MAX_M)
        self._simulator_action_space = self.action_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # The target patrol steers with a consistent heading convention (see
        # JetSim.update_enemy_ai). Set before any episode is used.
        self.sim.correct_enemy_heading = C.TARGET_PATROL_CORRECT_HEADING

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        scenario = (options or {}).get("scenario")
        if seed is not None:
            self._train_rng = np.random.default_rng(seed)
        if scenario is None:
            self._train_episode += 1
            scenario = training_scenario(self._train_rng, self._train_episode)
        elif isinstance(scenario, dict):
            scenario = Scenario(**{k: v for k, v in scenario.items()
                                   if k in Scenario.__dataclass_fields__})
        super().reset(seed=scenario.seed)
        self._apply_scenario(scenario)
        self.scenario = scenario
        self.launched = False
        self.launch = None
        self.outcome = None
        self.min_distance = self._distance_to_target()
        self.episode_return = 0.0
        self.trajectory = []
        obs = self._get_observation()
        self._record(np.zeros(4, np.float32), obs)
        return obs.astype(np.float32), {"scenario": scenario.as_dict()}

    def _apply_scenario(self, s):
        psi_p, psi_t = math.radians(s.player_heading), math.radians(s.target_heading)
        b = math.radians(s.player_heading + s.bearing)
        target = Vec3(s.range_m * math.sin(b), s.range_m * math.cos(b), s.altitude + s.delta_altitude)
        # Parent reset sets the target velocity from the compass heading ...
        self._reset_simulation_state({
            "scenario_id": s.scenario_id, "seed": s.seed, "enemy_initial_position": tuple(target),
            "enemy_initial_speed": s.target_speed, "enemy_initial_heading": s.target_heading,
            "enemy_initial_altitude": float(target.z)})
        enemy = self.sim.enemy
        enemy.yaw = -s.target_heading % 360.0   # ... and its nose must agree (Panda yaw is CCW).
        # The legacy reset copies ENEMY_START_HPR.x (= 180, a heading) into roll,
        # which clamps to a 90 deg bank. Start the target wings-level instead.
        enemy.roll = enemy.pitch = 0.0
        enemy.velocity = Vec3(math.sin(psi_t), math.cos(psi_t), 0) * s.target_speed
        player = self.sim.player
        player.position = Vec3(0, 0, s.altitude)
        player.roll = player.pitch = 0.0
        player.yaw = -s.player_heading % 360.0
        player.velocity = Vec3(math.sin(psi_p), math.cos(psi_p), 0) * s.player_speed
        player.current_speed = player.target_speed = player.airspeed = s.player_speed
        player.altitude = s.altitude
        for actor in (player, enemy):
            actor.sync_graphics()
        self._prev_range = self._distance_to_target()
        self._prev_cos_ata = self._current_cos_ata()

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(-1), -1, 1)
        if action.shape != (4,):
            raise ValueError(f"expected 4 flight actions, got {action.shape}")
        if self.outcome is not None:
            raise RuntimeError("step() called after the episode ended")
        prev_d, prev_cos = self._distance_to_target(), self._current_cos_ata()
        rel_start = np.array(self.sim.enemy.position - self.sim.player.position, float)
        self._pre_step = (np.array(self.sim.player.velocity, float), np.array(self.sim.enemy.velocity, float))
        launched_before = self.launched
        events_before = len(self.sim.missile_events)
        fire_request = 0.0 if launched_before else 1.0     # the fixed launch rule

        # The parent clips against its own 5-D (flight + fire) space.
        exposed, self.action_space = self.action_space, self._simulator_action_space
        try:
            obs, _, _, _, info = super().step(np.append(action, fire_request))
        finally:
            self.action_space = exposed

        d = self._distance_to_target()
        cos_ata = float(obs[11])
        rel_end = np.array(self.sim.enemy.position - self.sim.player.position, float)
        self.min_distance = min(self.min_distance, segment_min_distance(rel_start, rel_end))
        weapon = info["weapon"]
        if weapon["valid_fire_attempt"]:
            self.launched = True
            self.launch = {"step": self._episode_step, "time_s": self._episode_step * C.SIMULATION_DT,
                           "range_m": prev_d, "cos_ata": prev_cos, **launch_geometry(rel_start, self._pre_step)}

        outcome = self._outcome(weapon, events_before)
        terms = reward_terms(prev_d, d, prev_cos, cos_ata, float(obs[0]), float(obs[1]),
                             float(obs[2]), launched_before)
        terms["time"] = -C.TIME_PENALTY
        terms["launch"] = C.LAUNCH_REWARD if weapon["valid_fire_attempt"] else 0.0
        terms["terminal"] = {"hit": C.HIT_REWARD, "collision": C.COLLISION_PENALTY,
                             "ground": C.COLLISION_PENALTY,
                             "boundary": C.BOUNDARY_PENALTY}.get(outcome, 0.0)
        weights = {"closure": C.W_CLOSURE, "alignment": C.W_ALIGNMENT, "geometry": C.W_GEOMETRY,
                   "speed": C.W_SPEED, "altitude": C.W_ALTITUDE, "aoa": C.W_AOA}
        reward = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        self.episode_return += reward

        truncated = outcome is None and self._episode_step >= C.MAX_EPISODE_STEPS
        if truncated:
            outcome = "timeout"
        terminated = outcome is not None and not truncated
        self.outcome = outcome
        self._record(action, obs)
        out_info = {"reward_terms": terms, "range_m": d, "cos_ata": cos_ata, "launched": self.launched}
        if outcome is not None:
            out_info["episode_summary"] = self.summary()
            out_info["outcome"] = outcome          # read by the training Monitor
        return obs.astype(np.float32), float(reward), terminated, truncated, out_info

    def _outcome(self, weapon, events_before):
        if self.min_distance < C.COLLISION_RADIUS_M:
            return "collision"          # takes precedence over a same-step hit
        if self.sim.player.position.z < 0.0:
            return "ground"
        if weapon["missile_hit_this_step"]:
            return SUCCESS
        if self.sim.game_over and "CHASE BOUNDARY" in self.sim.game_over_message:
            return "boundary"
        for event in self.sim.missile_events[events_before:]:
            if event.team == "friendly" and event.reason != "hit":
                return "missile_" + str(event.reason)
        if self.launched and not self.sim.active_missiles and not self.sim.target_destroyed:
            return "missile_removed"
        return None

    # -------------------------------------------------------------- reporting
    def summary(self):
        launch = self.launch or {}
        event = next((e for e in self.sim.missile_events if e.team == "friendly"), None)
        return {"outcome": self.outcome, "success": self.outcome == SUCCESS,
                "steps": self._episode_step, "time_s": self._episode_step * C.SIMULATION_DT,
                "launched": self.launched, "launch_time_s": launch.get("time_s", math.nan),
                "launch_range_m": launch.get("range_m", math.nan),
                "launch_cos_ata": launch.get("cos_ata", math.nan),
                "launch_aspect_deg": launch.get("aspect_deg", math.nan),
                "launch_closing_speed_mps": launch.get("closing_speed_mps", math.nan),
                # From the simulator's own missile event (exact, not sampled at 0.05 s).
                "missile_closest_approach_m": event.closest_approach if event else math.nan,
                "missile_flight_time_s": event.age if event else math.nan,
                "missile_end_reason": event.reason if event else "",
                "min_distance_m": self.min_distance, "return": self.episode_return}

    def _record(self, action, obs):
        if not self.record_trajectory:
            return
        p, e = self.sim.player, self.sim.enemy
        missile = self.sim.active_missiles[0]["position"] if self.sim.active_missiles else (math.nan,) * 3
        self.trajectory.append({
            "t": self._episode_step * C.SIMULATION_DT,
            **{f"p_{k}": v for k, v in zip("xyz", p.position)},
            **{f"p_v{k}": v for k, v in zip("xyz", p.velocity)},
            "p_roll": p.roll, "p_pitch": p.pitch, "p_yaw": p.yaw, "p_speed": p.airspeed,
            **{f"t_{k}": v for k, v in zip("xyz", e.position)},
            **{f"t_v{k}": v for k, v in zip("xyz", e.velocity)},
            **{f"m_{k}": v for k, v in zip("xyz", missile)},
            "range_m": float(obs[10]), "cos_ata": float(obs[11]),
            "a_pitch": float(action[0]), "a_roll": float(action[1]),
            "a_yaw": float(action[2]), "a_throttle": float(action[3]),
            "launched": self.launched})
