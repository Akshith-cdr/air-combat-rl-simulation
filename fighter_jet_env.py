import os

os.environ.setdefault("PANDA_PRC", "window-type offscreen\nfullscreen 0\n")

from math import acos, atan2, cos, degrees, hypot, radians, sin

import gymnasium as gym
import numpy as np
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import Vec3

import updated_chase_jet_sim_v3 as sim_mod


OBSERVATION_ORDER = [
    "V",
    "h",
    "alpha",
    "nz",
    "phi",
    "theta",
    "psi",
    "delta_x",
    "delta_y",
    "delta_z",
    "d",
    "cos_ATA",
    "Vrel",
    "relative_vertical_velocity",
    "d_dot",
    "velocity_bearing_error",
    "velocity_elevation_error",
    "relative_lateral_velocity",
    "target_speed",
]

# Reward configuration. Values marked as placeholders are intentionally exposed for tuning in
# a later reward-design pass; they are not treated as finalized physics or MDP limits.
W_CLOSURE = 1.0
W_ALIGNMENT = 1.0
W_GEOMETRY = 1.0
W_SPEED = 0.5
W_ALTITUDE = 0.5
W_AOA = 1.0
W_BOUNDARY = 0.5
TERMINAL_SUCCESS_REWARD = 100.0
TERMINAL_BOUNDARY_PENALTY = -25.0
INVALID_FIRE_PENALTY = -1.0
VALID_FIRE_REWARD = 5.0
MISSILE_HIT_REWARD = TERMINAL_SUCCESS_REWARD
TIME_PENALTY = 0.01
CLOSURE_SCALE = 250.0
ALIGNMENT_SCALE = 0.01
D_GATE = 1000.0
GEOMETRY_PROGRESS_SCALE = 0.10
MIN_SAFE_SPEED = 150.0
MIN_SAFE_ALTITUDE = 250.0
MAX_SAFE_AOA = 15.0

# Broad placeholder bounds required for Gymnasium compatibility when exact simulator limits
# are not already codified in the MDP or runtime. These are intentionally conservative and
# wide so the environment can be instantiated without silently inventing a tighter domain.
OBSERVATION_LOW = np.array(
    [
        0.0,
        0.0,
        -90.0,
        0.0,
        -180.0,
        -90.0,
        0.0,
        -10000.0,
        -10000.0,
        -10000.0,
        0.0,
        -1.0,
        0.0,
        -1600.0,
        -1600.0,
        -1000.0,
        -180.0,
        -1600.0,
        0.0,
    ],
    dtype=np.float32,
)

OBSERVATION_HIGH = np.array(
    [
        1000.0,
        10000.0,
        90.0,
        20.0,
        180.0,
        90.0,
        360.0,
        10000.0,
        10000.0,
        10000.0,
        10000.0,
        1.0,
        1000.0,
        1600.0,
        1600.0,
        1000.0,
        180.0,
        1600.0,
        800.0,
    ],
    dtype=np.float32,
)


class FighterJetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomized_enemy_training=False, hybrid_fire=False):
        self.randomized_enemy_training = bool(randomized_enemy_training)
        self.hybrid_fire = bool(hybrid_fire)
        self._randomized_enemy_episode = 0
        self.sim = sim_mod.JetSim()
        self.action_space = gym.spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0, 0.0 if self.hybrid_fire else -1.0],
                dtype=np.float32,
            ),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            shape=(19,),
            dtype=np.float32,
        )
        self._prev_range = None
        self._prev_cos_ata = None
        self._last_dt = 1.0 / 60.0
        self._episode_step = 0
        self._episode_reward = 0.0
        self._previous_fire_signal = 0.0
        self.reset()

    def _sample_randomized_enemy_training_scenario(self, simulator_seed):
        """Sample one training-only scenario from the environment's seeded RNG."""
        self._randomized_enemy_episode += 1
        altitude = float(self.np_random.uniform(700.0, 2200.0))
        return {
            "scenario_id": f"TRAIN-{self._randomized_enemy_episode:08d}",
            "seed": int(simulator_seed),
            "enemy_initial_position": (
                float(self.np_random.uniform(-1200.0, 1200.0)),
                float(self.np_random.uniform(2200.0, 4800.0)),
                altitude,
            ),
            "enemy_initial_speed": float(self.np_random.uniform(140.0, 240.0)),
            "enemy_initial_heading": float(self.np_random.uniform(0.0, 360.0)),
            "enemy_initial_altitude": altitude,
        }

    def _set_player_controls(self, action):
        self.sim.player.pitch_input = float(action[0])
        self.sim.player.roll_input = float(action[1])
        self.sim.player.yaw_input = float(action[2])
        self.sim.player.speed_input = float(action[3])

    def _missile_available(self):
        """Return whether the simulator's single mounted missile is still present."""
        missile = getattr(self.sim, "missile", None)
        return missile is not None and not missile.isHidden()

    def _process_fire_signal(self, fire_signal, current_range, current_cos_ata):
        """Apply one fire attempt only on a non-positive-to-positive signal edge."""
        fire_request = fire_signal > 0.0 and self._previous_fire_signal <= 0.0
        self._previous_fire_signal = float(fire_signal)
        if not fire_request:
            return False, False, False

        valid_fire = (
            not getattr(self.sim, "target_destroyed", False)
            and not self.sim.game_over
            and self._missile_available()
            and current_range < 100.0
            and current_cos_ata > 0.95
        )
        if not valid_fire:
            return True, False, True

        active_before = len(self.sim.active_missiles)
        self.sim.fire_missile()
        launched = len(self.sim.active_missiles) > active_before
        return True, launched, not launched

    def _process_hybrid_fire(self, fire_bit, current_range, current_cos_ata):
        """Apply a hybrid-policy binary fire request as a one-step pulse."""
        fire_request = float(fire_bit) >= 0.5
        if not fire_request:
            return False, False, False

        valid_fire = (
            not getattr(self.sim, "target_destroyed", False)
            and not self.sim.game_over
            and self._missile_available()
            and current_range < 100.0
            and current_cos_ata > 0.95
        )
        if not valid_fire:
            return True, False, True

        active_before = len(self.sim.active_missiles)
        self.sim.fire_missile()
        launched = len(self.sim.active_missiles) > active_before
        return True, launched, not launched

    def _clear_missile_runtime_state(self):
        effects = getattr(self.sim, "missile_vfx", None)
        if effects is not None:
            effects.reset()
        if getattr(self.sim, "active_missiles", None):
            for missile_state in list(self.sim.active_missiles):
                node = missile_state.get("node")
                if node is not None:
                    try:
                        node.removeNode()
                    except Exception:
                        pass
            self.sim.active_missiles = []

        if self.sim.missile is not None:
            try:
                self.sim.missile.show()
            except Exception:
                pass

    def _reset_simulation_state(self, enemy_challenge_scenario=None):
        player = self.sim.player
        enemy = self.sim.enemy

        self._clear_missile_runtime_state()
        self.sim.missile_hit = False
        self.sim.missile_hit_count = 0
        self.sim.target_destroyed = False

        player.position = Vec3(sim_mod.PLAYER_START_POS)
        player.velocity = Vec3(0, 150, 0)
        player.roll = float(sim_mod.PLAYER_START_HPR.x)
        player.pitch = float(sim_mod.PLAYER_START_HPR.y)
        player.yaw = float(sim_mod.PLAYER_START_HPR.z)

        player.roll_rate = 0.0
        player.pitch_rate = 0.0
        player.yaw_rate = 0.0
        player.pitch_input = 0.0
        player.roll_input = 0.0
        player.yaw_input = 0.0
        player.speed_input = 0.0
        player.target_speed = 150.0
        player.current_speed = 150.0
        player.engine_on = True
        player.airspeed = 150.0
        player.altitude = max(0.0, player.position.z)
        player.g_force = 1.0
        player.turn_rate = 0.0
        player.target_turn_rate = 0.0
        player.crashed = False
        player.crash_message = ""

        if enemy_challenge_scenario is None:
            enemy_position = Vec3(sim_mod.ENEMY_START_POS)
            enemy_speed = 150.0
            enemy_heading = float(sim_mod.ENEMY_START_HPR.z)
            enemy_velocity = Vec3(0, enemy_speed, 0)
        else:
            required_fields = {
                "scenario_id",
                "seed",
                "enemy_initial_position",
                "enemy_initial_speed",
                "enemy_initial_heading",
                "enemy_initial_altitude",
            }
            missing_fields = required_fields.difference(enemy_challenge_scenario)
            if missing_fields:
                raise ValueError(
                    "Enemy challenge scenario is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            position_values = tuple(enemy_challenge_scenario["enemy_initial_position"])
            if len(position_values) != 3:
                raise ValueError("enemy_initial_position must contain exactly three values")

            position_values = tuple(float(value) for value in position_values)
            enemy_altitude = float(enemy_challenge_scenario["enemy_initial_altitude"])
            enemy_speed = float(enemy_challenge_scenario["enemy_initial_speed"])
            enemy_heading = float(enemy_challenge_scenario["enemy_initial_heading"]) % 360.0
            numeric_values = (
                *position_values,
                enemy_altitude,
                enemy_speed,
                enemy_heading,
            )
            if not all(np.isfinite(value) for value in numeric_values):
                raise ValueError("Enemy challenge scenario values must be finite")
            if abs(position_values[2] - enemy_altitude) > 1e-6:
                raise ValueError(
                    "enemy_initial_position z must equal enemy_initial_altitude"
                )
            if not 0.0 <= enemy_speed <= sim_mod.MAX_SPEED:
                raise ValueError(
                    f"enemy_initial_speed must be within 0..{sim_mod.MAX_SPEED} m/s"
                )

            enemy_position = Vec3(*position_values)
            heading_radians = radians(enemy_heading)
            enemy_velocity = Vec3(
                sin(heading_radians) * enemy_speed,
                cos(heading_radians) * enemy_speed,
                0.0,
            )

        enemy.position = enemy_position
        enemy.velocity = enemy_velocity
        enemy.roll = float(sim_mod.ENEMY_START_HPR.x)
        enemy.pitch = float(sim_mod.ENEMY_START_HPR.y)
        enemy.yaw = enemy_heading

        enemy.roll_rate = 0.0
        enemy.pitch_rate = 0.0
        enemy.yaw_rate = 0.0
        enemy.pitch_input = 0.0
        enemy.roll_input = 0.0
        enemy.yaw_input = 0.0
        enemy.speed_input = 0.0
        enemy.target_speed = enemy_speed
        enemy.current_speed = enemy_speed
        enemy.engine_on = True
        enemy.airspeed = enemy_speed
        enemy.altitude = max(0.0, enemy.position.z)
        enemy.g_force = 1.0
        enemy.turn_rate = 0.0
        enemy.target_turn_rate = 0.0
        enemy.crashed = False
        enemy.crash_message = ""
        enemy.autopilot = True
        enemy.ai_patrol_center = Vec3(enemy_position)
        enemy.ai_waypoint = Vec3(enemy_position)
        enemy.ai_update_timer = 0.0

        self.sim.game_over = False
        self.sim.game_over_message = ""
        self.sim.update_jet_position(player)
        self.sim.update_jet_position(enemy)
        if self.sim.enemy.boundary_sphere is not None:
            self.sim.enemy.boundary_sphere.setPos(0, 0, 0)
            self.sim.enemy.boundary_sphere.setScale(sim_mod.CHASE_BOUNDARY_RADIUS)

        self._prev_range = float((enemy.position - player.position).length())
        self._prev_cos_ata = self._current_cos_ata()
        self._last_dt = 1.0 / 60.0

    def _distance_to_target(self):
        rel = self.sim.player.position - self.sim.enemy.position
        return float(rel.length())

    def _current_cos_ata(self):
        player = self.sim.player
        rel = self.sim.enemy.position - self.sim.player.position
        if player.velocity.length() <= 1e-8 or rel.length() <= 1e-8:
            return 0.0
        velocity_dir = player.velocity.normalized()
        los_dir = rel.normalized()
        return float(np.clip(velocity_dir.dot(los_dir), -1.0, 1.0))

    def _compute_current_aoa(self):
        player = self.sim.player
        velocity = player.velocity
        forward = player.get_forward_vector()

        if velocity.length() > 1e-8:
            velocity_dir = velocity.normalized()
        else:
            velocity_dir = forward

        try:
            dot = np.clip(velocity_dir.dot(forward), -1.0, 1.0)
            aoa = degrees(acos(float(dot)))
            if velocity_dir.z < forward.z:
                aoa = -aoa
            return float(aoa)
        except Exception:
            return float(player.pitch)

    @staticmethod
    def _heading_degrees(vector):
        return float(degrees(atan2(vector.x, vector.y)) % 360.0)

    @staticmethod
    def _elevation_degrees(vector):
        horizontal_magnitude = hypot(vector.x, vector.y)
        return float(degrees(atan2(vector.z, max(1e-8, horizontal_magnitude))))

    def _get_observation(self):
        player = self.sim.player
        enemy = self.sim.enemy

        rel = enemy.position - player.position
        d = float(rel.length())
        if self._prev_range is not None and self._last_dt > 0:
            d_dot = (d - self._prev_range) / self._last_dt
        else:
            d_dot = 0.0

        velocity = player.velocity
        if velocity.length() > 1e-8:
            velocity_dir = velocity.normalized()
        else:
            velocity_dir = player.get_forward_vector()

        los = rel
        if los.length() > 1e-8:
            los = los.normalized()
        else:
            los = Vec3(0, 1, 0)

        cos_ata = float(np.clip(velocity_dir.dot(los), -1.0, 1.0))
        relative_velocity = enemy.velocity - player.velocity
        v_rel = float(relative_velocity.length())

        target_bearing = self._heading_degrees(rel)
        player_velocity_heading = self._heading_degrees(player.velocity)
        velocity_bearing_error = float(
            sim_mod.wrap_angle_deg(target_bearing - player_velocity_heading)
        )

        target_los_elevation = self._elevation_degrees(rel)
        player_velocity_elevation = self._elevation_degrees(player.velocity)
        velocity_elevation_error = float(
            target_los_elevation - player_velocity_elevation
        )

        horizontal_range = hypot(rel.x, rel.y)
        if horizontal_range > 1e-8:
            horizontal_right_of_los = Vec3(
                rel.y / horizontal_range,
                -rel.x / horizontal_range,
                0.0,
            )
            relative_lateral_velocity = float(
                relative_velocity.dot(horizontal_right_of_los)
            )
        else:
            relative_lateral_velocity = 0.0

        obs = np.array(
            [
                float(player.airspeed),
                float(player.altitude),
                float(self._compute_current_aoa()),
                float(player.g_force),
                float(player.roll),
                float(player.pitch),
                float(player.yaw),
                float(rel.x),
                float(rel.y),
                float(rel.z),
                float(d),
                float(cos_ata),
                float(v_rel),
                float(relative_velocity.z),
                float(d_dot),
                velocity_bearing_error,
                velocity_elevation_error,
                relative_lateral_velocity,
                float(player.target_speed),
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward_components(
        self, prev_range, prev_cos_ata, obs_current, dt, weapon_reward=0.0
    ):
        player = self.sim.player
        enemy = self.sim.enemy
        current_range = float(obs_current[10])
        current_cos_ata = float(obs_current[11])
        current_d_dot = float(obs_current[14])
        current_speed = float(obs_current[0])
        current_altitude = float(obs_current[1])
        current_aoa = float(obs_current[2])

        # Engagement terms.
        r_closure = float(
            np.clip(
                (prev_range - current_range) / (CLOSURE_SCALE * (1.0 / 60.0)),
                -1.0,
                1.0,
            )
        )
        r_alignment = float(
            max(0.0, r_closure)
            * max(
                0.0,
                np.clip(
                    (current_cos_ata - 0.95) / (1.0 - 0.95),
                    -1.0,
                    1.0,
                ),
            )
        )
        # Near-terminal joint progress toward the firing envelope. Both states
        # use the same clipped potential, so exiting the gate is penalized.
        current_distance_progress = float(
            np.clip(
                (D_GATE - current_range) / (D_GATE - 100.0),
                0.0,
                1.0,
            )
        )
        previous_distance_progress = float(
            np.clip(
                (D_GATE - prev_range) / (D_GATE - 100.0),
                0.0,
                1.0,
            )
        )
        current_alignment_progress = float(
            np.clip((current_cos_ata + 1.0) / 2.0, 0.0, 1.0)
        )
        previous_alignment_progress = float(
            np.clip((prev_cos_ata + 1.0) / 2.0, 0.0, 1.0)
        )
        phi_current = current_distance_progress * current_alignment_progress
        phi_previous = previous_distance_progress * previous_alignment_progress
        r_geometry = float(
            np.clip(
                (phi_current - phi_previous) / GEOMETRY_PROGRESS_SCALE,
                -1.0,
                1.0,
            )
        )

        # Flight terms.
        if current_speed >= MIN_SAFE_SPEED:
            r_speed = 0.0
        else:
            r_speed = -float(np.clip((MIN_SAFE_SPEED - current_speed) / max(1.0, MIN_SAFE_SPEED * 0.5), 0.0, 1.0))

        if current_altitude >= MIN_SAFE_ALTITUDE:
            r_altitude = 0.0
        else:
            r_altitude = -float(np.clip((MIN_SAFE_ALTITUDE - current_altitude) / max(1.0, MIN_SAFE_ALTITUDE * 0.5), 0.0, 1.0))

        aoa_abs = abs(current_aoa)
        if aoa_abs <= MAX_SAFE_AOA:
            r_aoa = 0.0
        else:
            r_aoa = -float(np.clip((aoa_abs - MAX_SAFE_AOA) / 30.0, 0.0, 1.0))

        boundary_margin = sim_mod.CHASE_BOUNDARY_RADIUS - current_range
        if boundary_margin > 0.5 * sim_mod.CHASE_BOUNDARY_RADIUS:
            r_boundary = 0.0
        else:
            r_boundary = -float(np.clip((0.5 * sim_mod.CHASE_BOUNDARY_RADIUS - boundary_margin) / (0.5 * sim_mod.CHASE_BOUNDARY_RADIUS), 0.0, 1.0))

        r_weapon = float(weapon_reward)

        # Time penalty is treated as a small negative constant.
        r_time = -TIME_PENALTY

        engagement = W_CLOSURE * r_closure + W_ALIGNMENT * r_alignment + W_GEOMETRY * r_geometry
        flight = W_SPEED * r_speed + W_ALTITUDE * r_altitude + W_AOA * r_aoa + W_BOUNDARY * r_boundary

        terminal = 0.0
        if self.sim.game_over and "CHASE BOUNDARY" in self.sim.game_over_message:
            terminal = TERMINAL_BOUNDARY_PENALTY

        total = terminal + engagement + flight + r_weapon + r_time

        return {
            "terminal": terminal,
            "closure": r_closure,
            "alignment": r_alignment,
            "geometry": r_geometry,
            "speed": r_speed,
            "altitude": r_altitude,
            "aoa": r_aoa,
            "boundary": r_boundary,
            "weapon": r_weapon,
            "time": r_time,
            "total": total,
            "engagement": engagement,
            "flight": flight,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        simulator_seed = seed if seed is not None else int(self.np_random.integers(0, 2**32))
        self.sim.set_seed(simulator_seed)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._previous_fire_signal = 0.0
        enemy_challenge_scenario = (
            options.get("enemy_challenge_scenario") if options is not None else None
        )
        if enemy_challenge_scenario is None and self.randomized_enemy_training:
            enemy_challenge_scenario = self._sample_randomized_enemy_training_scenario(
                simulator_seed
            )
        self._reset_simulation_state(enemy_challenge_scenario)
        self._prev_range = float((self.sim.enemy.position - self.sim.player.position).length())
        self._prev_cos_ata = self._current_cos_ata()
        self._last_dt = 1.0 / 60.0
        obs = self._get_observation()
        info = {"episode": {"length": 0, "terminated": False, "truncated": False}}
        if enemy_challenge_scenario is not None:
            info["challenge_scenario_id"] = enemy_challenge_scenario["scenario_id"]
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 5:
            raise ValueError(f"Expected action with 5 values, got {action.size}")

        prev_range = self._distance_to_target()
        prev_cos_ata = self._current_cos_ata()
        self._prev_range = prev_range
        self._prev_cos_ata = prev_cos_ata

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_player_controls(clipped_action)
        fire_processor = (
            self._process_hybrid_fire if self.hybrid_fire else self._process_fire_signal
        )
        fire_request, valid_fire_attempt, invalid_fire_attempt = fire_processor(
            float(clipped_action[4]), prev_range, prev_cos_ata
        )
        missile_hit_before_step = bool(getattr(self.sim, "missile_hit", False))

        dt = float(globalClock.getDt())
        dt = min(max(dt, 0.0), 0.05)
        self._last_dt = dt

        if not self.sim.game_over:
            self.sim.update_enemy_ai(dt)
            self.sim.update_actor_physics(self.sim.player, dt)
            self.sim.update_actor_physics(self.sim.enemy, dt)
            self.sim.check_boundary()

        self.sim.update_jet_position(self.sim.player)
        self.sim.update_jet_position(self.sim.enemy)
        if self.sim.enemy.boundary_sphere is not None:
            self.sim.enemy.boundary_sphere.setPos(0, 0, 0)
            self.sim.enemy.boundary_sphere.setScale(sim_mod.CHASE_BOUNDARY_RADIUS)

        self.sim.update_missiles(dt)
        self.sim.update_camera()
        self.sim.update_hud()

        obs = self._get_observation()
        self._episode_step += 1

        missile_hit_this_step = (
            not missile_hit_before_step and bool(getattr(self.sim, "missile_hit", False))
        )
        weapon_reward = (
            (INVALID_FIRE_PENALTY if invalid_fire_attempt else 0.0)
            + (VALID_FIRE_REWARD if valid_fire_attempt else 0.0)
            + (MISSILE_HIT_REWARD if missile_hit_this_step else 0.0)
        )
        reward_components = self._compute_reward_components(
            prev_range, prev_cos_ata, obs, dt, weapon_reward
        )
        reward = float(reward_components["total"])
        terminated = bool(self.sim.game_over) or missile_hit_this_step
        truncated = False  # Placeholder until a formal truncation rule is defined.
        self._episode_reward += reward

        self._prev_range = float((self.sim.enemy.position - self.sim.player.position).length())
        self._prev_cos_ata = self._current_cos_ata()

        terminal_reason = ""
        if missile_hit_this_step:
            terminal_reason = "missile_hit"
        elif self.sim.game_over and "CHASE BOUNDARY" in self.sim.game_over_message:
            terminal_reason = self.sim.game_over_message

        info = {
            "episode": {
                "r": self._episode_reward,
                "l": self._episode_step,
                "length": self._episode_step,
                "terminated": terminated,
                "truncated": truncated,
            },
            "terminal_reason": terminal_reason,
            "dt": dt,
            "range_to_target": self._distance_to_target(),
            "weapon": {
                "fire_request": fire_request,
                "valid_fire_attempt": valid_fire_attempt,
                "invalid_fire_attempt": invalid_fire_attempt,
                "missile_available": self._missile_available(),
                "missile_hit": bool(getattr(self.sim, "missile_hit", False)),
                "missile_hit_this_step": missile_hit_this_step,
                "target_destroyed": bool(getattr(self.sim, "target_destroyed", False)),
            },
            "reward_breakdown": {
                "terminal": reward_components["terminal"],
                "closure": reward_components["closure"],
                "alignment": reward_components["alignment"],
                "geometry": reward_components["geometry"],
                "speed": reward_components["speed"],
                "altitude": reward_components["altitude"],
                "aoa": reward_components["aoa"],
                "boundary": reward_components["boundary"],
                "weapon": reward_components["weapon"],
                "time": reward_components["time"],
                "engagement": reward_components["engagement"],
                "flight": reward_components["flight"],
                "total": reward,
            },
        }

        return obs.astype(np.float32), reward, terminated, truncated, info


__all__ = ["FighterJetEnv"]
