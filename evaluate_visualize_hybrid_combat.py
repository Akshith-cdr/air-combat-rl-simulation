"""Evaluate viewer-configured episodes until simulator combat outcomes occur.

Flight and firing both come from the selected policy. The policy's fire action is
passed unchanged to the environment, which launches only inside the
500 m <= range <= 1000 m, cos(ATA) > 0.95 envelope (weapon_config.py).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import updated_chase_jet_sim_v3 as sim_mod
from fighter_jet_env import DEFAULT_SIMULATION_DT
from hybrid_ppo import HybridActorCriticPolicy
from test_final import DETAIL_COLUMNS
import visualize_hybrid_live as live_viewer
from weapon_config import (DEFAULT_MISSILE_LAUNCH_CEILING_M, MAX_MISSILE_LAUNCH_DISTANCE_M,
                           MIN_MISSILE_LAUNCH_DISTANCE_M)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = live_viewer.DEFAULT_MODEL
VECNORMALIZE_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_vecnormalize.pkl")
CONFIG_PATH = MODEL_PATH.parents[1] / "config.json"
BUNDLE_PATH = MODEL_PATH.with_name("bundle.json")
DEFAULT_EPISODES = 1000
# Episodes that reach this many 0.05 s steps (100 s, the training horizon)
# without a hit, boundary exit or missile loss end as timeouts.
MAX_EPISODE_STEPS = 2000
LAUNCH_ENVELOPE = (f"{MIN_MISSILE_LAUNCH_DISTANCE_M:g} m <= range <= {MAX_MISSILE_LAUNCH_DISTANCE_M:g} m, "
                   "cos(ATA) > 0.95")
DEFAULT_BASE_SEED = 987654321
# "deterministic": fire when p(fire) >= 0.5. "sampled": the fire bit is sampled from the
# policy's Bernoulli (seeded per episode); flight stays at the Gaussian mean (Stage 7 only).
FIRE_MODE = "sampled"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_seeds(base_seed: int, count: int = DEFAULT_EPISODES) -> list[int]:
    """Unique uint32 seeds, stable for a base-seed prefix."""
    if not 0 <= base_seed < 2**64 or count <= 0:
        raise ValueError("Base seed must be uint64 and episode count must be positive")
    seed = int(np.random.SeedSequence(base_seed, spawn_key=(0,)).generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(seed)
    result: list[int] = []
    used: set[int] = set()
    while len(result) < count:
        candidate = int(rng.integers(0, 2**32, dtype=np.uint32))
        if candidate not in used:
            used.add(candidate)
            result.append(candidate)
    return result


def input_paths() -> dict[str, Path]:
    return {
        "checkpoint": MODEL_PATH,
        "vecnormalize": VECNORMALIZE_PATH,
        "flight_config": CONFIG_PATH,
        "checkpoint_bundle": BUNDLE_PATH,
    }


def verify_inputs() -> dict[str, str]:
    paths = input_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing visualization combat input(s): " + ", ".join(missing))
    hashes = {str(path.resolve()): sha256(path) for path in paths.values()}
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    bundle_hashes = bundle.get("hashes", {})
    for name, path in ((MODEL_PATH.name, MODEL_PATH), (VECNORMALIZE_PATH.name, VECNORMALIZE_PATH)):
        if bundle_hashes.get(name) != hashes[str(path.resolve())]:
            raise ValueError(f"Checkpoint bundle does not pin the selected model/statistics pair: {path}")
    return hashes


def make_environment(stats_path: Path = VECNORMALIZE_PATH) -> VecNormalize:
    """Use the viewer's environment factory; headless changes presentation only."""
    return live_viewer.make_live_environment(
        stats_path,
        guided_missiles=True,
        radar_range_m=2000.0,
        visible_missiles=False,
        launch_range_m=DEFAULT_MISSILE_LAUNCH_CEILING_M,
        headless=True,
    )


def validate_model(model: PPO, environment: VecNormalize) -> None:
    policy = model.policy
    if not isinstance(policy, HybridActorCriticPolicy):
        raise TypeError("Selected checkpoint is not a hybrid flight/fire controller")
    if model.observation_space.shape != (19,) or model.action_space.shape != (5,):
        raise ValueError("Viewer controller requires 19 observations and 5 actions")
    if tuple(policy.action_net.weight.shape) != (4, 64) or tuple(policy.fire_net.weight.shape) != (1, 64):
        raise ValueError("Checkpoint does not have the expected continuous-flight and fire heads")
    if environment.observation_space != model.observation_space or environment.action_space != model.action_space:
        raise ValueError("VecNormalize spaces do not match the selected checkpoint")
    if (not environment.norm_obs or environment.norm_reward
            or environment.obs_rms.mean.shape != (19,) or environment.obs_rms.count <= 0
            or not np.isfinite(environment.obs_rms.mean).all()
            or not np.isfinite(environment.obs_rms.var).all()):
        raise ValueError("Unexpected or invalid VecNormalize statistics")


def raw_observation(environment: VecNormalize, done: bool, info: dict[str, Any]) -> np.ndarray:
    if done and info.get("terminal_observation") is not None:
        return np.asarray(environment.unnormalize_obs(info["terminal_observation"]), dtype=np.float64)
    return np.asarray(environment.get_original_obs()[0], dtype=np.float64).copy()


def geometry(raw: np.ndarray) -> tuple[float, float, float, float]:
    dx, dy, dz = (float(value) for value in raw[7:10])
    return math.sqrt(dx * dx + dy * dy + dz * dz), math.hypot(dx, dy), abs(dz), float(raw[11])


def fire_authority(model: PPO) -> str:
    """Name the component that produces the policy's fire action."""
    adapter = getattr(model.policy, "fire_adapter", None)
    if adapter is not None and getattr(adapter, "mode", "policy") != "policy":
        return f"embedded {adapter.mode} fire classifier (Stage 6 checkpoint)"
    if FIRE_MODE == "sampled":
        return "PPO Bernoulli fire head (fire bit sampled, seeded per episode; flight at Gaussian mean)"
    return "PPO Bernoulli fire head (deterministic: fire when p >= 0.5)"


def fire_probability(model: PPO, observation: np.ndarray) -> float:
    tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        distribution = model.policy.get_distribution(tensor)
        bernoulli = getattr(distribution, "bernoulli", None)
        if bernoulli is not None:
            return float(bernoulli.probs.detach().cpu().reshape(-1)[0])
        # FixedConjunctionFlightPolicy exposes the deterministic decision as a
        # point mass; report its actual controller output as 0/1.
        action = model.predict(observation, deterministic=True)[0]
        return float(np.asarray(action).reshape(-1)[4])


def actual_initial_state(environment: VecNormalize) -> dict[str, float]:
    raw_env = getattr(environment.venv.envs[0], "unwrapped", environment.venv.envs[0])
    player, enemy = raw_env.sim.player, raw_env.sim.enemy
    relative = enemy.position - player.position
    return {
        "ucav_initial_x": float(player.position.x), "ucav_initial_y": float(player.position.y),
        "ucav_initial_z": float(player.position.z), "ucav_initial_speed": float(player.velocity.length()),
        "ucav_initial_heading": float(player.yaw), "enemy_initial_x": float(enemy.position.x),
        "enemy_initial_y": float(enemy.position.y), "enemy_initial_z": float(enemy.position.z),
        "enemy_initial_speed": float(enemy.velocity.length()), "enemy_initial_heading": float(enemy.yaw),
        "initial_3d_distance": float(relative.length()),
    }


def viewer_episode_configuration(simulation_seed: int) -> tuple[str, int]:
    """Resolve the same REC scenario and policy seed as a viewer CLI launch."""
    scenario_id, resolved_seed, policy_seed, _ = live_viewer.resolve_launch_configuration(
        None, simulation_seed, None
    )
    if resolved_seed != simulation_seed:
        raise AssertionError("Explicit viewer simulation seed must be preserved")
    return scenario_id, policy_seed


def no_missile_available(raw_env) -> bool:
    """True only when the simulator has neither stored ammunition nor an in-flight round."""
    return not raw_env._missile_available() and not raw_env.sim.active_missiles


def evaluate_episode(environment: VecNormalize, model: PPO, episode_number: int, seed: int) -> dict[str, Any]:
    scenario_id, policy_seed = viewer_episode_configuration(seed)
    reset_scenario = live_viewer.asdict(live_viewer.scenario_by_id(scenario_id))
    reset_scenario["seed"] = int(seed)
    # This is the viewer CLI's post-load policy seed. It is retained even
    # though its default deterministic policy mode does not sample actions.
    np.random.seed(policy_seed)
    th.manual_seed(policy_seed)
    environment.set_options({"enemy_challenge_scenario": reset_scenario})
    environment.seed(seed)
    observation = environment.reset()
    raw_env = getattr(environment.venv.envs[0], "unwrapped", environment.venv.envs[0])
    result: dict[str, Any] = {"scenario_id": scenario_id, "scenario_seed": seed,
                              **actual_initial_state(environment)}
    raw = raw_observation(environment, False, {})
    distance, horizontal, vertical, cos_ata = geometry(raw)
    elapsed = total_reward = 0.0
    minimum_3d, minimum_horizontal, minimum_vertical = distance, horizontal, vertical
    maximum_cos_ata = cos_ata
    minimum_valid_fire_range = maximum_fire_probability = math.nan
    time_to_minimum, time_to_valid_fire = 0.0, math.nan
    time_to_launch = time_to_hit = math.nan
    requests = valid_requests = invalid_requests = launches = hits = 0
    terminal_reason, length = "", 0

    while True:
        length += 1
        pre_step_distance = float(raw[10])
        probability = fire_probability(model, observation)
        if math.isnan(maximum_fire_probability) or probability > maximum_fire_probability:
            maximum_fire_probability = probability
        action, _ = model.predict(observation, deterministic=FIRE_MODE == "deterministic")
        action_values = np.asarray(action).reshape(-1)
        if action_values.shape != (5,) or action_values[4] not in (0.0, 1.0):
            raise ValueError("Hybrid policy must emit a five-element action with binary fire output")
        missile_event_offset = len(raw_env.sim.missile_events)
        observation, reward, done_array, infos = environment.step(action)
        done, info = bool(done_array[0]), infos[0]
        dt = float(info.get("dt", math.nan))
        if not math.isfinite(dt) or not math.isclose(dt, DEFAULT_SIMULATION_DT, abs_tol=1e-12):
            raise ValueError("Environment returned an invalid simulation timestep")
        elapsed += dt
        total_reward += float(reward[0])
        weapon = info.get("weapon", {})
        valid_fire = bool(weapon.get("valid_fire_attempt", False))
        invalid_fire = bool(weapon.get("invalid_fire_attempt", False))
        hit_this_step = bool(weapon.get("missile_hit_this_step", False))
        missile_events = raw_env.sim.missile_events[missile_event_offset:]
        requests += int(bool(weapon.get("fire_request", False)))
        valid_requests += int(valid_fire)
        invalid_requests += int(invalid_fire)
        if valid_fire:
            launches += 1
            minimum_valid_fire_range = pre_step_distance if math.isnan(minimum_valid_fire_range) else min(minimum_valid_fire_range, pre_step_distance)
            if math.isnan(time_to_valid_fire): time_to_valid_fire = elapsed
            if math.isnan(time_to_launch): time_to_launch = elapsed
        if hit_this_step:
            hits += 1
            if math.isnan(time_to_hit): time_to_hit = elapsed
        raw = raw_observation(environment, done, info)
        distance, horizontal, vertical, cos_ata = geometry(raw)
        if distance < minimum_3d:
            minimum_3d, time_to_minimum = distance, elapsed
        minimum_horizontal = min(minimum_horizontal, horizontal)
        minimum_vertical = min(minimum_vertical, vertical)
        maximum_cos_ata = max(maximum_cos_ata, cos_ata)
        if done:
            terminal_reason = str(info.get("terminal_reason", "") or "terminated")
            if terminal_reason != "missile_hit" and "CHASE BOUNDARY" not in terminal_reason.upper():
                raise RuntimeError(f"Unexpected environment terminal outcome: {terminal_reason}")
            break
        # A non-hit terminal event is the simulator's own record that this
        # launched guided missile was removed without destroying its target.
        wasted = [event for event in missile_events
                  if event.team == "friendly" and event.target_id == "enemy"
                  and event.reason != "hit"]
        if wasted:
            terminal_reason = "missile_wasted:" + wasted[0].reason
            break
        if no_missile_available(raw_env):
            terminal_reason = "no_missile_available"
            break
        if length >= MAX_EPISODE_STEPS:
            terminal_reason = "evaluation_step_limit"
            break

    result.update({
        # Success requires both the environment's missile-hit terminal event
        # and the recorded actual missile-hit event. Geometry alone never counts.
        "success": terminal_reason == "missile_hit" and hits > 0,
        "terminal_reason": terminal_reason,
        "missile_hit": hits > 0,
        "boundary_failure": "CHASE BOUNDARY" in terminal_reason.upper(),
        "timeout": terminal_reason == "evaluation_step_limit",
        "episode_length_steps": length, "episode_time_seconds": elapsed, "total_reward": total_reward,
        "minimum_3d_distance_m": minimum_3d, "minimum_horizontal_distance_m": minimum_horizontal,
        "minimum_vertical_distance_m": minimum_vertical, "maximum_cosATA": maximum_cos_ata,
        "minimum_range_at_valid_fire": minimum_valid_fire_range,
        "maximum_fire_action_or_fire_probability_if_available": maximum_fire_probability,
        "number_of_fire_attempts": requests, "number_of_valid_fire_attempts": valid_requests,
        "number_of_invalid_fire_attempts": invalid_requests, "missile_launch_count": launches,
        "missile_hit_count": hits, "time_to_min_distance_seconds": time_to_minimum,
        "time_to_first_valid_fire_seconds": time_to_valid_fire, "time_to_missile_launch_seconds": time_to_launch,
        "time_to_missile_hit_seconds": time_to_hit, "final_3d_distance_m": distance,
        "final_cosATA": cos_ata, "final_d_dot": float(raw[14]), "final_speed": float(raw[0]),
    })
    return result


def termination_condition(row: dict[str, Any]) -> str:
    reason = str(row["terminal_reason"])
    if reason == "missile_hit":
        return "missile_hit"
    if "CHASE BOUNDARY" in reason.upper():
        return "chase_boundary"
    if reason.startswith("missile_wasted:"):
        return "missile_wasted"
    if reason == "no_missile_available":
        return "no_missile_available"
    if reason == "evaluation_step_limit":
        return "timeout"
    raise ValueError(f"Unexpected evaluator terminal reason: {reason}")


def summarize(rows: list[dict[str, Any]], base_seed: int, input_hashes: dict[str, str],
              requested_episodes: int = DEFAULT_EPISODES, authority: str = "") -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "schema": "visualize-hybrid-combat-evaluation-v6", "requested_episodes": requested_episodes,
        "completed_episodes": len(rows), "base_seed": base_seed, "policy_mode": FIRE_MODE,
        "checkpoint_path": str(MODEL_PATH.resolve()), "vecnormalize_path": str(VECNORMALIZE_PATH.resolve()),
        "flight_config_path": str(CONFIG_PATH.resolve()), "checkpoint_bundle_path": str(BUNDLE_PATH.resolve()),
        "input_sha256": input_hashes,
        "environment": {
            "simulation_dt_seconds": DEFAULT_SIMULATION_DT, "guided_missiles": True,
            "missile_guidance": "PN + LOS acquisition", "launch_range_ceiling_m": DEFAULT_MISSILE_LAUNCH_CEILING_M,
            "launch_envelope": LAUNCH_ENVELOPE, "max_episode_steps": MAX_EPISODE_STEPS,
            "episode_termination": {
                "success": "environment missile_hit_this_step with missile_hit terminal reason",
                "boundary_failure": "existing environment chase-boundary terminal reason",
                "missile_wasted": "existing non-hit guided MissileEvent from the simulator",
                "no_missile_available": "simulator reports no mounted missile and no active projectile",
                "timeout": f"{MAX_EPISODE_STEPS} steps without any of the above",
            },
            "randomized_enemy_training": False,
            "spawn_mechanism": "viewer resolve_launch_configuration(seed): seed-selected REC-001..REC-006 reset scenario",
            "enemy_controller": "JetSim.update_enemy_ai rule-based patrol",
            "patrol_waypoint_selection": {name: getattr(sim_mod, name) for name in (
                "ENEMY_PATROL_RADIUS", "ENEMY_PATROL_VERTICAL_RADIUS", "ENEMY_PATROL_MIN_ALTITUDE", "ENEMY_PATROL_MAX_ALTITUDE")},
            "firing_authority": authority,
        },
        "missile_hit_successes": successes,
        "missile_hit_success_rate": successes / len(rows) if rows else None,
        "non_hit_episodes": len(rows) - successes,
        "terminal_reason_counts": dict(Counter(str(row["terminal_reason"]) for row in rows)),
        "termination_condition_counts": dict(Counter(termination_condition(row) for row in rows)),
        "total_missile_hits": sum(int(row["missile_hit_count"]) for row in rows),
        "total_fire_attempts": sum(int(row["number_of_fire_attempts"]) for row in rows),
        "total_valid_fire_attempts": sum(int(row["number_of_valid_fire_attempts"]) for row in rows),
        "total_invalid_fire_attempts": sum(int(row["number_of_invalid_fire_attempts"]) for row in rows),
        "total_missile_launches": sum(int(row["missile_launch_count"]) for row in rows),
    }


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_model(model_path: Path) -> None:
    """Evaluate another checkpoint, e.g. experiments/fire_policy/<run-id>/final/candidate.zip."""
    global MODEL_PATH, VECNORMALIZE_PATH, CONFIG_PATH, BUNDLE_PATH
    MODEL_PATH = Path(model_path).resolve()
    VECNORMALIZE_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_vecnormalize.pkl")
    CONFIG_PATH = MODEL_PATH.parents[1] / "config.json"
    BUNDLE_PATH = MODEL_PATH.with_name("bundle.json")


def evaluation_manifest(base_seed: int, input_hashes: dict[str, str],
                        episodes: int = DEFAULT_EPISODES) -> dict[str, Any]:
    """The exact configuration required to resume a partial evaluation."""
    return {
        "schema": "visualize-hybrid-combat-evaluation-v6",
        "requested_episodes": episodes, "base_seed": base_seed,
        "episode_seeds_path": "seed_list.json",
        "model_path": str(MODEL_PATH.resolve()), "vecnormalize_path": str(VECNORMALIZE_PATH.resolve()),
        "flight_config_path": str(CONFIG_PATH.resolve()), "checkpoint_bundle_path": str(BUNDLE_PATH.resolve()),
        "input_sha256": input_hashes, "detail_columns": DETAIL_COLUMNS,
        "policy_mode": FIRE_MODE,
        "environment": {"randomized_enemy_training": False, "firing": "policy fire action, unchanged",
                        "guided_missiles": True, "launch_range_ceiling_m": DEFAULT_MISSILE_LAUNCH_CEILING_M,
                        "launch_envelope": LAUNCH_ENVELOPE, "max_episode_steps": MAX_EPISODE_STEPS,
                        "simulation_dt_seconds": DEFAULT_SIMULATION_DT,
                        "episode_termination": "environment missile hit, chase boundary, non-hit guided MissileEvent, no mounted/active missile, or step limit",
                        "scenario_selection": "resolve_launch_configuration(None, episode_seed, None)"},
    }


def load_completed_rows(detail_path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Read only complete, schema-valid rows written by this evaluator."""
    with detail_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != DETAIL_COLUMNS:
            raise ValueError("Existing CSV schema does not match the required evaluation schema")
        rows = list(reader)
    if len(rows) > len(seeds):
        raise ValueError("Existing CSV has more rows than the requested 1,000 episodes")
    int_fields = {"scenario_seed", "missile_hit_count", "number_of_fire_attempts",
                  "number_of_valid_fire_attempts", "number_of_invalid_fire_attempts", "missile_launch_count"}
    for index, row in enumerate(rows, 1):
        if set(row) != set(DETAIL_COLUMNS) or any(value is None for value in row.values()):
            raise ValueError(f"Existing CSV row {index} is incomplete")
        expected_id, expected_seed = viewer_episode_configuration(seeds[index - 1])[0], seeds[index - 1]
        if row["scenario_id"] != expected_id or int(row["scenario_seed"]) != expected_seed:
            raise ValueError(f"Existing CSV row {index} does not match the recorded seed schedule")
        row["success"] = row["success"] == "True"
        for name in int_fields:
            row[name] = int(row[name])
    return rows


def prepare_output(output: Path, manifest: dict[str, Any], seeds: list[int]) -> tuple[Path, Path, list[dict[str, Any]], bool]:
    """Create a new output or safely resume a matching interrupted evaluation."""
    detail_path, summary_path = output / f"final_eval_{len(seeds)}.csv", output / "summary.json"
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        save_json_atomic(output / "manifest.json", manifest)
        save_json_atomic(output / "seed_list.json", {"episode_seeds": seeds})
        return detail_path, summary_path, [], False
    if not output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output}")
    manifest_path, seeds_path = output / "manifest.json", output / "seed_list.json"
    required = (manifest_path, seeds_path, detail_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError("Existing output cannot be resumed because it lacks: " + ", ".join(missing))
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if existing_manifest != manifest or existing_seeds != {"episode_seeds": seeds}:
        raise ValueError(
            "Existing output uses a different evaluator configuration and cannot be resumed. "
            "Its files are preserved; choose a new --output directory."
        )
    rows = load_completed_rows(detail_path, seeds)
    return detail_path, summary_path, rows, True


def run(output: Path, episodes: int = DEFAULT_EPISODES, base_seed: int = DEFAULT_BASE_SEED) -> tuple[Path, Path]:
    # The seed list is prefix-stable: --episodes N runs the first N episodes of
    # the 1,000-episode evaluation, so its rows can be compared with the full CSV.
    if not 1 <= episodes <= DEFAULT_EPISODES:
        raise ValueError("episodes must be between 1 and 1,000")
    input_hashes = verify_inputs()
    seeds = episode_seeds(base_seed, episodes)
    manifest = evaluation_manifest(base_seed, input_hashes, episodes)
    detail_path, summary_path, rows, resumed = prepare_output(output, manifest, seeds)
    print(f"Model checkpoint loaded: {MODEL_PATH.resolve()}")
    print(f"VecNormalize statistics loaded: {VECNORMALIZE_PATH.resolve()}")
    print(f"Flight configuration loaded: {CONFIG_PATH.resolve()}")
    print(f"Checkpoint bundle loaded: {BUNDLE_PATH.resolve()}")
    print(f"Launch envelope: {LAUNCH_ENVELOPE}")
    print(f"Episode termination: missile hit, chase boundary, simulator missile-waste event, "
          f"no remaining missile, or {MAX_EPISODE_STEPS} steps")
    print(f"{'Resuming' if resumed else 'Writing'} {episodes} episode rows at: {detail_path.resolve()}")
    print(f"Completed rows already present: {len(rows)}")

    environment: VecNormalize | None = None
    authority = ""
    if len(rows) == episodes:
        summary = summarize(rows, base_seed, input_hashes, episodes)
        summary["run_status"] = "completed"
        save_json_atomic(summary_path, summary)
        print(f"Evaluation already completed. Summary: {summary_path.resolve()}")
        return detail_path, summary_path
    initial_summary = summarize(rows, base_seed, input_hashes, episodes)
    initial_summary["run_status"] = "in_progress"
    save_json_atomic(summary_path, initial_summary)
    try:
        environment = make_environment(VECNORMALIZE_PATH)
        model = PPO.load(str(MODEL_PATH), env=environment, device="cpu")
        validate_model(model, environment)
        if FIRE_MODE == "sampled" and getattr(model.policy, "train_flight", True):
            raise ValueError("--fire-mode sampled needs a Stage 7 policy with frozen, mean-flown flight")
        authority = fire_authority(model)
        print(f"Fire authority: {authority}")
        with detail_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=DETAIL_COLUMNS, extrasaction="raise")
            if not resumed:
                writer.writeheader()
                stream.flush(); os.fsync(stream.fileno())
            for episode_number, seed in enumerate(seeds[len(rows):], len(rows) + 1):
                row = evaluate_episode(environment, model, episode_number, seed)
                writer.writerow(row)
                stream.flush(); os.fsync(stream.fileno())
                rows.append(row)
                if episode_number == 1 or episode_number % 10 == 0 or episode_number == episodes:
                    print(f"Episode {episode_number}/{episodes}: {row['terminal_reason']}", flush=True)
                progress = summarize(rows, base_seed, input_hashes, episodes, authority)
                progress["run_status"] = "in_progress"
                save_json_atomic(output / "progress.json", progress)
                save_json_atomic(summary_path, progress)
    except BaseException as exc:
        partial = summarize(rows, base_seed, input_hashes, episodes, authority)
        partial["run_status"] = "interrupted_or_failed"
        partial["error_type"] = type(exc).__name__
        partial["error_message"] = str(exc)
        save_json_atomic(summary_path, partial)
        raise
    finally:
        if environment is not None:
            try:
                raw_env = getattr(environment.venv.envs[0], "unwrapped", environment.venv.envs[0])
                raw_env.sim.destroy()
            finally:
                environment.close()
    summary = summarize(rows, base_seed, input_hashes, episodes, authority)
    summary["run_status"] = "completed"
    save_json_atomic(summary_path, summary)
    print(f"Completed {len(rows)}/{episodes} episodes. Summary: {summary_path.resolve()}")
    return detail_path, summary_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> None:
    global FIRE_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--base-seed", type=nonnegative_int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="Checkpoint (candidate.zip) inside a Stage 6/7 run; default: the Stage 7 final model")
    parser.add_argument("--episodes", type=positive_int, default=DEFAULT_EPISODES,
                        help="Run only the first N of the 1,000 seeded episodes (quick check)")
    parser.add_argument("--fire-mode", choices=("deterministic", "sampled"), default=FIRE_MODE,
                        help="sampled (default): sample the learned fire decision, seeded per episode; deterministic: fire only when p >= 0.5")
    args = parser.parse_args(argv)
    FIRE_MODE = args.fire_mode
    select_model(args.model_path)
    run(args.output, args.episodes, args.base_seed)


if __name__ == "__main__":
    main()
