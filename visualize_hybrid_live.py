"""Onscreen trained hybrid PPO viewer: learned flight and learned firing."""

import argparse
import sys
import os
import secrets
import random
import time
import math
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

# FighterJetEnv uses setdefault(), so this must precede every project import.
os.environ["PANDA_PRC"] = (
    "window-type onscreen\n"
    "win-size 1280 720\n"
    "show-frame-rate-meter 1\n"
)

import numpy as np
import torch as th  
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, loadPrcFileData
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluate_ppo import FROZEN_ENEMY_CHALLENGE_MANIFEST, MAX_EVALUATION_STEPS
from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy
from live_visualization import PreserveTerminalVisuals, PhysicsFrameBuffer
from live_diagnostics import FlightDiagnostics
from radar import RadarConfig
from weapon_config import MIN_MISSILE_LAUNCH_DISTANCE_M, DEFAULT_MISSILE_LAUNCH_CEILING_M
from visualization_telemetry import combat_telemetry

loadPrcFileData("hybrid_live", "win-size 1280 720\nshow-frame-rate-meter 1\n")


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_MODEL = PROJECT_ROOT / "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
# Stage 7 output: flight and fire decisions both come from this PPO policy.
DEFAULT_MODEL = PROJECT_ROOT / "experiments/fire_policy/stage7_fire/final/candidate.zip"
DEFAULT_SCENARIO = None
RANDOM_SCENARIOS = tuple(f"REC-{index:03d}" for index in range(1, 7))


def normalization_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def scenario_by_id(scenario_id: str):
    matches = [
        scenario
        for scenario in FROZEN_ENEMY_CHALLENGE_MANIFEST
        if scenario.scenario_id == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate challenge scenario: {scenario_id}")
    return matches[0]


def resolve_simulation_seed(requested_seed: int | None) -> int:
    """Use an explicit seed or draw a fresh 32-bit seed from the operating system."""
    return requested_seed if requested_seed is not None else secrets.randbits(32)


def resolve_launch_configuration(scenario_id, simulation_seed, policy_seed):
    """Fresh OS entropy by default; one supplied environment seed replays defaults.

    Use a local RNG so choosing a scenario never consumes policy/simulator RNG.
    Keep scenario geometry intact rather than inventing unvalidated perturbations.
    """
    for label, value in (("Simulation", simulation_seed), ("Policy", policy_seed)):
        if value is not None and not 0 <= value < 2**32:
            raise ValueError(f"{label} seed must be in 0..4294967295")
    seed = resolve_simulation_seed(simulation_seed)
    generator = random.Random(seed)
    generated_scenario = generator.choice(RANDOM_SCENARIOS)
    generated_policy_seed = generator.getrandbits(32)
    scenario = generated_scenario if scenario_id is None else scenario_id
    scenario_by_id(scenario)  # Resolve against the actual supported manifest.
    sources = {
        "scenario": "generated from simulation seed" if scenario_id is None else "explicit",
        "simulation_seed": "generated from OS randomness" if simulation_seed is None else "explicit",
        "policy_seed": "generated from simulation seed" if policy_seed is None else "explicit",
    }
    return scenario, seed, generated_policy_seed if policy_seed is None else policy_seed, sources


def log_launch_configuration(arguments):
    print(f"Model checkpoint: {arguments.model_path.resolve()}")
    print(f"Selected scenario: {arguments.scenario} ({arguments.launch_sources['scenario']})")
    print(f"Simulation seed: {arguments.seed} ({arguments.launch_sources['simulation_seed']})")
    print(f"Policy seed: {arguments.policy_seed} ({arguments.launch_sources['policy_seed']})")
    print(f"Scenario/seed replay arguments: --scenario {arguments.scenario} "
          f"--seed {arguments.seed} --policy-seed {arguments.policy_seed}")


def configure_live_view(sim, radar_range_m=2000.0, visible_missiles=False):
    """One combat radar, optional simple glyphs, and one player camera."""
    sim.configure_presentation(missiles=False, lock_player_camera=True)
    if not sim.headless:
        # The environment is the sole actuation/physics owner in this viewer.
        for event in ("arrow_up", "arrow_down", "arrow_left", "arrow_right", "q", "e", "w", "s"):
            sim.ignore(event)
            sim.ignore(event + "-up")
        for event in ("m", "t", "c"):
            sim.ignore(event)
        sim.taskMgr.remove("update_task")
    simple = getattr(sim, "simple_missile_visuals", None)
    if simple is not None:
        simple.destroy()
    sim.simple_missile_visuals = None
    if visible_missiles and not sim.headless:
        from missile_vfx import SimpleMissilePresentation
        sim.simple_missile_visuals = SimpleMissilePresentation(sim.render, sim.load_missile_presentation_model())
    sim.radar_config = RadarConfig(range_m=radar_range_m)
    sim.update_radar()


def make_live_environment(stats_path: Path, guided_missiles=False, radar_range_m=2000.0,
                          visible_missiles=False, launch_range_m=DEFAULT_MISSILE_LAUNCH_CEILING_M,
                          headless=False):
    """The policy's own fire bit is passed unchanged to the environment's launch interlock."""
    def create_environment():
        raw = FighterJetEnv(hybrid_fire=True, missile_guidance=guided_missiles,
                            launch_range_gate=launch_range_m, headless=headless)
        configure_live_view(raw.sim, radar_range_m, visible_missiles)
        return PreserveTerminalVisuals(raw)

    vector_env = DummyVecEnv(
        [create_environment]
    )
    env = VecNormalize.load(str(stats_path), vector_env)
    env.action_space = vector_env.action_space
    env.training = False
    env.norm_reward = False
    return env


class PlaybackSchedule:
    """Wall-clock pacing only; never change the simulator's integration dt."""

    def __init__(self, rate=1.0, clock=None, sleep=None):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("playback_rate must be finite and positive")
        self.rate = rate
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.deadline = self.clock()

    def frames(self, dt, frame_delay):
        return max(1, math.ceil(dt / self.rate / (frame_delay if frame_delay > 0 else 1 / 60)))

    def pace(self, dt, frames):
        self.deadline += dt / self.rate / frames
        remaining = self.deadline - self.clock()
        if remaining > 0:
            self.sleep(remaining)
        elif remaining < -dt / self.rate:
            self.deadline = self.clock()


def missile_state(raw_env, weapon_info, launched: bool) -> str:
    if weapon_info.get("missile_hit_this_step") or weapon_info.get("missile_hit"):
        return "HIT"
    if raw_env.sim.active_missiles:
        return "IN FLIGHT"
    if raw_env._missile_available():
        return "READY"
    return "EXPENDED" if launched else "UNAVAILABLE"


def fire_state(fire_bit: int, weapon_info) -> str:
    if weapon_info.get("valid_fire_attempt"):
        return "1 — VALID LAUNCH"
    if weapon_info.get("invalid_fire_attempt"):
        return "1 — INVALID REQUEST"
    return f"{fire_bit} — {'REQUEST' if fire_bit else 'NO REQUEST'}"


def update_event_hud(hud, weapon_info):
    """A launch, expiry or removal must never manufacture a hit indication."""
    if weapon_info.get("missile_hit_this_step", False):
        message = "TARGET HIT"
        if weapon_info.get("target_destroyed", False):
            message += "\nTARGET DESTROYED"
        hud.setText(message)
    elif weapon_info.get("valid_fire_attempt", False):
        hud.setText("MISSILE AWAY")


def update_hud(
    hud,
    scenario_id,
    action_mode,
    step,
    simulation_time,
    raw_observation,
    fire_text,
    missile_text,
    terminal_reason,
    controller="Hybrid PPO",
    player=None,
    policy_action=None,
    guidance="PN + LOS acquisition",
    launch_range_m=DEFAULT_MISSILE_LAUNCH_CEILING_M,
    telemetry=None,
):
    if telemetry is not None:
        t = telemetry
        authority = "PPO FLIGHT / PPO FIRING"
        firing = f"PPO FIRE: {fire_text}"
        launch_band = f"{MIN_MISSILE_LAUNCH_DISTANCE_M:g} <= R <= {launch_range_m:g} m"
        hud.setText(
            f"{authority}\n"
            f"PLAYER: {t['player_speed_m_s']:.1f} m/s / {t['player_speed_km_h']:.0f} km/h\n"
            f"ENEMY: {t['enemy_speed_m_s']:.1f} m/s / {t['enemy_speed_km_h']:.0f} km/h\n"
            f"DISTANCE: {t['distance_m']:.1f} m   LAUNCH: {launch_band}\n"
            f"HEADING: {t['player_heading_deg']:.1f} deg  RATE: {t['player_heading_rate_deg_s']:+.1f} deg/s\n"
            f"SIM: {simulation_time:.2f} s  PLAYBACK: {t['requested_playback_rate']:g}x "
            f"/ {t['achieved_playback_rate']:.2f}x achieved\n"
            f"{firing}\nMISSILE: {missile_text}\n"
            f"{scenario_id} / {action_mode} / {terminal_reason or 'RUNNING'}"
        )
        return
    telemetry = ""
    if player is not None and policy_action is not None:
        telemetry = (f"PLAYER XYZ: {player.position.x:.1f}, {player.position.y:.1f}, {player.position.z:.1f} m\n"
                     f"SPEED: {player.velocity.length():.1f} m/s   PPO ACTION: "
                     + ", ".join(f"{value:+.2f}" for value in policy_action) + "\n")
    hud.setText(
        f"CONTROLLER: {controller}\n"
        f"ENEMY: rule-based patrol   GUIDANCE: {guidance}\n"
        f"LAUNCH GATE: {MIN_MISSILE_LAUNCH_DISTANCE_M:g} <= R <= {launch_range_m:g} m\n" +
        f"MODE: {action_mode}\n"
        f"SCENARIO: {scenario_id}\n"
        f"STEP: {step}   TIME: {simulation_time:6.2f} s\n"
        f"RANGE: {float(raw_observation[10]):8.2f} m\n"
        f"cos(ATA): {float(raw_observation[11]):+.5f}\n"
        f"FIRE: {fire_text}\n"
        f"MISSILE: {missile_text}\n"
        + telemetry +
        f"TERMINAL: {terminal_reason or 'RUNNING'}"
    )


def render_aircraft_frame(sim, frame_buffer=None, alpha=1.0):
    """Render actual scene samples with the player camera; never advance physics."""
    sim.update_camera()
    metrics = getattr(sim, "live_diagnostics", None)
    try:
        if frame_buffer is not None:
            frame_buffer.apply(alpha)
        with metrics.measure("window_events") if metrics is not None else nullcontext():
            if sim.win is not None and hasattr(sim.win, "processEvents"):
                sim.win.processEvents()
            sim.eventMgr.doEvents()
        with metrics.measure("graphics_render") if metrics is not None else nullcontext():
            sim.graphicsEngine.renderFrame()
        if metrics is not None:
            metrics.record_frame(sim, frame_buffer, alpha)
    finally:
        if frame_buffer is not None:
            frame_buffer.restore()


def visual_demo_scenario(seed, separation_m=1800.0):
    """Separate initial scenario; trained PPO still owns every player decision."""
    if not 500 <= separation_m <= 8000 or not math.isfinite(separation_m):
        raise ValueError("demo separation must be within 500..8000 metres")
    return {"scenario_id": "RL-SCENARIO-DEMO", "seed": int(seed),
            "enemy_initial_position": (150.0, float(separation_m), 1100.0),
            "enemy_initial_speed": 150.0, "enemy_initial_heading": 355.0,
            "enemy_initial_altitude": 1100.0}


def save_snapshot(sim, directory, label):
    if directory is not None:
        from panda3d.core import Filename
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sim.win.saveScreenshot(Filename.fromOsSpecific(str(directory / f"{label}.png")))


def hold_terminal_frame(sim, seconds: float, frame_delay: float):
    deadline = time.monotonic() + max(0.0, seconds)
    refresh_delay = frame_delay if frame_delay > 0.0 else 1.0 / 60.0
    while time.monotonic() < deadline:
        render_aircraft_frame(sim)
        time.sleep(refresh_delay)


def hold_terminal_presentation(sim, seconds: float, frame_delay: float):
    """Simple terminal review; all numerical simulation state remains stopped."""
    hold_terminal_frame(sim, seconds, frame_delay)


def run_live(
    model_path: Path,
    scenario_id: str,
    simulation_seed: int | None,
    policy_seed: int,
    deterministic: bool,
    max_steps: int,
    frame_delay: float,
    result_hold_seconds: float,
    radar_range_m: float = 2000.0,
    guided_missiles: bool = False,
    snapshot_dir: Path | None = None,
    visible_missiles: bool = True,
    visual_demo: bool = False,
    demo_separation_m: float = 1800.0,
    diagnostics_json: Path | None = None,
    launch_range_m: float = DEFAULT_MISSILE_LAUNCH_CEILING_M,
    combat_presentation: bool = False,
    playback_rate: float = 1.0,
    terminal_hold_seconds: float = 4.0,
    reseed_policy_after_load: bool = False,
):
    if not np.isfinite(radar_range_m) or radar_range_m <= 0:
        raise ValueError("Radar range must be finite and positive")
    if max_steps <= 0 or not math.isfinite(frame_delay) or frame_delay < 0:
        raise ValueError("max_steps must be positive and frame_delay finite and non-negative")
    if not math.isfinite(result_hold_seconds) or result_hold_seconds < 0:
        raise ValueError("result_hold_seconds must be finite and non-negative")
    if not math.isfinite(playback_rate) or playback_rate <= 0:
        raise ValueError("playback_rate must be finite and positive")
    if not math.isfinite(terminal_hold_seconds) or terminal_hold_seconds < 0:
        raise ValueError("terminal_hold_seconds must be finite and non-negative")
    if combat_presentation and not visible_missiles:
        raise ValueError("Combat presentation requires visible missiles")
    hold_seconds = terminal_hold_seconds if combat_presentation else result_hold_seconds
    stats_path = normalization_path(model_path)
    missing = [str(path) for path in (model_path, stats_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing live-visualization input: " + ", ".join(missing))
    seed = resolve_simulation_seed(simulation_seed)
    reset_scenario = (visual_demo_scenario(seed, demo_separation_m) if visual_demo else
                      asdict(scenario_by_id(scenario_id)))
    scenario_id = reset_scenario["scenario_id"]
    print(f"Simulation seed: {seed}")

    np.random.seed(policy_seed)
    th.manual_seed(policy_seed)
    if visual_demo:
        guided_missiles = True
    env = make_live_environment(stats_path, guided_missiles, radar_range_m,
                                visible_missiles, launch_range_m)
    print(f"Weapon mode: {'guided (changed engagement dynamics)' if guided_missiles else 'legacy'}")
    hud = None
    event_hud = None
    metrics = FlightDiagnostics() if diagnostics_json is not None else None
    pending_images = []

    def measure(name):
        return metrics.measure(name) if metrics is not None else nullcontext()

    def queue_image(label):
        if snapshot_dir is not None:
            # Capture only; expensive PNG encoding/file writes happen after playback.
            with measure("snapshot_capture"):
                pending_images.append((label, sim.win.getScreenshot()))

    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        checkpoint_rng_seed = int(th.initial_seed())
        if reseed_policy_after_load:
            # SB3 load restores its saved seed. CLI launches must honor their
            # resolved seed; existing programmatic replay callers stay intact.
            np.random.seed(policy_seed)
            th.manual_seed(policy_seed)
        if not isinstance(model.policy, HybridActorCriticPolicy):
            raise TypeError("The supplied checkpoint is not a hybrid PPO policy")
        if model is not None and (model.observation_space.shape != (19,) or model.action_space.shape != (5,)):
            raise ValueError("Live hybrid visualization requires 19 observations and 5 actions")
        if model is not None and tuple(model.policy.action_net.weight.shape) != (4, 64):
            raise ValueError("Expected four continuous Gaussian action means")
        if model is not None and tuple(model.policy.log_std.shape) != (4,):
            raise ValueError("Expected four continuous Gaussian log standard deviations")
        if model is not None and tuple(model.policy.fire_net.weight.shape) != (1, 64):
            raise ValueError("Expected one Bernoulli fire logit")

        reset_scenario["seed"] = int(seed)
        env.set_options({"enemy_challenge_scenario": reset_scenario})
        env.seed(seed)
        observation = env.reset()
        vector = env.venv if hasattr(env, "venv") else env
        raw_env = vector.envs[0].unwrapped
        sim = raw_env.sim
        if metrics is not None:
            metrics.attach(sim)
            metrics.observe_environment(raw_env)
            metrics.metadata = {
                "model_path": str(model_path.resolve()), "statistics_path": str(stats_path.resolve()),
                "hashes_before": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (model_path, stats_path)},
                "policy_class": type(model.policy).__name__, "training_timesteps_metadata": model.num_timesteps,
                "requested_policy_seed_before_load": policy_seed,
                "torch_initial_seed_after_load": int(th.initial_seed()),
                "checkpoint_rng_seed": checkpoint_rng_seed,
                "policy_seed_reapplied_after_load": reseed_policy_after_load,
                "controller": "trained PPO", "enemy_controller": "JetSim.update_enemy_ai: rule-based patrol",
                "missile_controller": "PN + LOS acquisition" if guided_missiles else "legacy linear",
                "separate_initial_scenario": visual_demo, "scenario": reset_scenario,
                "combat_presentation": combat_presentation, "playback_rate": playback_rate,
                "terminal_hold_seconds": hold_seconds,
                "launch_range_gate_m": raw_env.launch_range_gate,
                "reward_geometry_reference_range_m": raw_env.fire_range_gate,
                "statistics_training": env.training, "norm_reward": env.norm_reward,
                "normalization_epsilon": env.epsilon, "clip_obs": env.clip_obs, "norm_obs": env.norm_obs,
                "statistics_before": {"mean": env.obs_rms.mean.tolist(), "var": env.obs_rms.var.tolist(),
                                      "count": float(env.obs_rms.count)}}
        sim.hud_text.hide()
        hud = OnscreenText(
            text="",
            pos=(-1.29, 0.91),
            scale=0.041,
            align=TextNode.ALeft,
            fg=(1.0, 1.0, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            mayChange=True,
        )
        event_hud = OnscreenText(
            text="",
            pos=(0.0, 0.70),
            scale=0.085,
            align=TextNode.ACenter,
            fg=(1.0, 0.86, 0.08, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.95),
            mayChange=True,
        )

        mode = ("RL INITIAL SCENARIO / " if visual_demo else "") + ("DETERMINISTIC FIRE" if deterministic else "SAMPLED FIRE")
        simulation_time = 0.0
        launched = False
        terminal_reason = ""
        last_fire_text = "0 — NO REQUEST"
        last_missile_text = "READY" if raw_env._missile_available() else "UNAVAILABLE"
        print(f"Trained PPO live visualization: model={model_path}, vecnormalize={stats_path}, "
              f"scenario={scenario_id}, simulation_seed={seed}, "
              f"requested_preload_policy_seed={policy_seed}, torch_seed_after_load={th.initial_seed()}, "
              f"mode={mode.lower()}; enemy=rule-based patrol, guidance=conventional")

        frame_buffer = PhysicsFrameBuffer(sim)
        # Compile the normal aircraft scene before the paced physics loop.
        if metrics is not None:
            metrics.phase = "warmup"
        with measure("render_frame"):
            render_aircraft_frame(sim)
        if sim.simple_missile_visuals is not None:
            with measure("missile_model_resource_warmup"):
                sim.simple_missile_visuals.prepare_resources(sim.camera, lambda: render_aircraft_frame(sim))
        if metrics is not None:
            metrics.phase = "episode"
        playback = PlaybackSchedule(playback_rate)
        episode_wall_start = time.monotonic()
        telemetry = None
        for step in range(1, max_steps + 1):
            if metrics is not None:
                metrics.step = step
            frame_buffer.before_step()
            with measure("policy_inference"):
                action = model.predict(observation, deterministic=deterministic)[0]
            if metrics is not None:
                metrics.record_prediction(observation, action, env.get_original_obs()[0])
            action_values = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_values.shape != (5,):
                raise ValueError(f"Unexpected hybrid action shape: {action_values.shape}")
            if action_values[4] not in (0.0, 1.0):
                raise ValueError(f"Hybrid fire output is not binary: {action_values[4]}")

            # The entire policy vector, including its fire bit, goes to the environment.
            previous_heading = float(sim.player.yaw)
            with measure("simulation_step"):
                observation, _, done, infos = env.step(action)
            if metrics is not None:
                metrics.record_step()
            with measure("presentation_sync"):
                frame_buffer.after_step()
            info = infos[0]
            weapon_info = info.get("weapon", {})
            simulation_time += float(info.get("dt", 0.0))
            dt = float(info.get("dt", raw_env.simulation_dt))
            if combat_presentation:
                telemetry = combat_telemetry(sim, previous_heading, dt, simulation_time,
                    playback_rate, time.monotonic() - episode_wall_start)
                if metrics is not None:
                    metrics.speed_telemetry.append(dict(step=step, **telemetry))
            launched |= bool(weapon_info.get("valid_fire_attempt", False))
            last_fire_text = fire_state(int(action_values[4]), weapon_info)
            last_missile_text = missile_state(raw_env, weapon_info, launched)
            terminal_reason = (
                info.get("terminal_reason", "") or "environment_terminated"
                if bool(done[0])
                else ""
            )
            raw_observation = env.get_original_obs()[0].copy()
            update_event_hud(event_hud, weapon_info)

            if weapon_info.get("valid_fire_attempt", False):
                print(
                    f"Real missile launch: step={step}, t={simulation_time:.3f}s, "
                    f"range={raw_observation[10]:.3f}m, cosATA={raw_observation[11]:.6f}, "
                    f"active_projectiles={len(sim.active_missiles)}"
                )
            if weapon_info.get("missile_hit_this_step", False):
                print(
                    f"Real missile hit: step={step}, t={simulation_time:.3f}s, "
                    f"hit_count={sim.missile_hit_count}"
                )

            update_hud(
                hud,
                scenario_id,
                mode,
                step,
                simulation_time,
                raw_observation,
                last_fire_text,
                last_missile_text,
                terminal_reason,
                controller="Trained Hybrid PPO",
                player=sim.player, policy_action=action_values,
                guidance="PN + LOS acquisition" if guided_missiles else "legacy linear",
                launch_range_m=raw_env.launch_range_gate,
                telemetry=telemetry,
            )
            dt = float(info.get("dt", raw_env.simulation_dt))
            frames = playback.frames(dt, frame_delay)
            for sample in range(1, frames + 1):
                with measure("render_frame"):
                    render_aircraft_frame(sim, frame_buffer=frame_buffer, alpha=sample / frames)
                if frame_delay > 0:
                    playback.pace(dt, frames)
            if step == 1:
                queue_image("initial")
            if weapon_info.get("valid_fire_attempt", False):
                queue_image("launch")
            if sim.active_missiles and step % 20 == 0:
                queue_image(f"flight-{step:04d}")
            if bool(done[0]):
                queue_image("terminal")

            if bool(done[0]):
                print(f"Live episode ended at step {step}: {terminal_reason}")
                if combat_presentation or playback_rate != 1.0:
                    elapsed = max(1e-9, time.monotonic() - episode_wall_start)
                    print(f"Playback: requested {playback_rate:g}x, achieved {simulation_time / elapsed:.2f}x")
                if metrics is not None:
                    metrics.phase = "terminal_review"
                if hold_seconds > 0:
                    event_hud.setText(f"EPISODE ENDED: {terminal_reason}\nRequested terminal review; physics stopped")
                    with measure("terminal_hold"):
                        if combat_presentation:
                            hold_terminal_presentation(sim, hold_seconds, frame_delay)
                        else:
                            hold_terminal_frame(sim, hold_seconds, frame_delay)
                return terminal_reason

        terminal_reason = "visualization_step_limit"
        update_hud(
            hud,
            scenario_id,
            mode,
            max_steps,
            simulation_time,
            raw_env._get_observation(),
            last_fire_text,
            last_missile_text,
            terminal_reason,
            telemetry=telemetry,
            launch_range_m=raw_env.launch_range_gate,
        )
        print(f"Live episode reached the {max_steps}-step safety limit")
        if combat_presentation or playback_rate != 1.0:
            elapsed = max(1e-9, time.monotonic() - episode_wall_start)
            print(f"Playback: requested {playback_rate:g}x, achieved {simulation_time / elapsed:.2f}x")
        if hold_seconds > 0:
            if combat_presentation:
                hold_terminal_presentation(sim, hold_seconds, frame_delay)
            else:
                hold_terminal_frame(sim, hold_seconds, frame_delay)
        return terminal_reason
    finally:
        if hud is not None:
            hud.destroy()
        if event_hud is not None:
            event_hud.destroy()
        env.close()
        if snapshot_dir is not None:
            from panda3d.core import Filename
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            for label, image in pending_images:
                image.write(Filename.fromOsSpecific(str(Path(snapshot_dir) / f"{label}.png")))
        if metrics is not None:
            metrics.metadata["hashes_after"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in (model_path, stats_path)}
            metrics.metadata["statistics_after"] = {
                "mean": env.obs_rms.mean.tolist(), "var": env.obs_rms.var.tolist(),
                "count": float(env.obs_rms.count)}
            Path(diagnostics_json).parent.mkdir(parents=True, exist_ok=True)
            metrics.write(diagnostics_json)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-observation', action='store_true',
                        help='Separate learned camera-sighting task with frozen flight controls; no weapon actions.')
    parser.add_argument('--camera-run', type=Path,
                        default=PROJECT_ROOT/'camera_observation_runs/run_20261102_01',
                        help='Completed camera-only run containing camera_policy.zip and its manifest.')
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="Explicit scenario; otherwise choose REC-001..REC-006 from the simulation seed.")
    parser.add_argument("--radar-range-m", "--nav-range-m", type=float, default=2000.0,
                        help="Combat radar radius in metres (default: 2000); +X right/+Y up.")
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Optionally save initial, launch and terminal verification screenshots.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Simulation seed (default: fresh OS randomness); also reproduces omitted scenario/policy-seed defaults.",
    )
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="Explicit policy seed; otherwise generate reproducibly from the simulation seed.")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-delay", type=float, default=1.0 / 60.0)
    parser.add_argument("--combat-presentation", action=argparse.BooleanOptionalAction, default=True,
                        help="Simple numerical missile model, player camera and compact speed/combat HUD; no cinematic VFX.")
    parser.add_argument("--playback-rate", type=float, default=1.0,
                        help="Simulation seconds per wall second (e.g. 1.25); physics dt and actions stay unchanged. Frame-delay 0 is unpaced.")
    parser.add_argument("--terminal-hold-seconds", type=float, default=4.0,
                        help="Combat mode only: static terminal review with physics stopped (default 4; 0 disables).")
    parser.add_argument("--result-hold-seconds", type=float, default=0.0,
                        help="Normal mode only: explicit frozen terminal review (default 0). Combat mode uses --terminal-hold-seconds.")
    presentation = parser.add_mutually_exclusive_group()
    presentation.add_argument("--no-missile-vfx", "--no-vfx", action="store_true", default=False,
                              help="Radar-only mode; no missile glyphs or effects.")
    presentation.add_argument("--visible-missiles", dest="no_missile_vfx", action="store_false",
                              help="Lightweight numerical missile glyphs; no particles/trails/camera cuts.")
    parser.add_argument("--rl-scenario-demo", "--visual-demo", dest="visual_demo", action="store_true",
                        help="Separate initial scenario with trained PPO decisions and normal fire gate; no forced launch.")
    parser.add_argument("--demo-separation-m", type=float, default=1800.0)
    parser.add_argument("--diagnostics-json", type=Path,
                        help="Write per-step, integration, radar, render timings and flight distances after playback.")
    parser.add_argument("--guided-missiles", action=argparse.BooleanOptionalAction, default=True,
                        help="PN guidance (default enabled); engagement outcomes differ from legacy evaluation.")
    parser.add_argument("--no-cinematic-camera", action="store_true", default=True,
                        help="Legacy compatibility flag; the player camera remains locked in all viewer modes.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction, default=False,
        help="Fire only when p(fire) >= 0.5. Default: sample the learned fire decision "
             "(Stage 7 flight always uses the Gaussian mean), as in evaluation.",
    )
    return parser


def parse_live_arguments(argv=None):
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(tokens)
    if not arguments.camera_observation and any(token.split('=', 1)[0] == '--camera-run' for token in tokens):
        parser.error('--camera-run requires --camera-observation')
    if arguments.camera_observation:
        incompatible = {'--rl-scenario-demo', '--visual-demo', '--model-path'}
        supplied = {token.split('=', 1)[0] for token in tokens if token.startswith('--')}
        if supplied & incompatible:
            parser.error('Camera observation uses its own checkpoint and no weapon/demo controls')
        if (arguments.max_steps <= 0 or not math.isfinite(arguments.frame_delay) or arguments.frame_delay < 0
                or not math.isfinite(arguments.radar_range_m) or arguments.radar_range_m <= 0
                or not math.isfinite(arguments.playback_rate) or arguments.playback_rate <= 0
                or not math.isfinite(arguments.terminal_hold_seconds) or arguments.terminal_hold_seconds < 0):
            parser.error('Invalid camera viewer timing, horizon or radar range')
        try:
            arguments.seed = resolve_simulation_seed(arguments.seed)
        except ValueError as exc:
            parser.error(str(exc))
        return arguments
    try:
        arguments.scenario, arguments.seed, arguments.policy_seed, arguments.launch_sources = (
            resolve_launch_configuration(arguments.scenario, arguments.seed, arguments.policy_seed))
    except ValueError as exc:
        parser.error(str(exc))
    return arguments


def main(argv=None):
    arguments = parse_live_arguments(argv)
    if arguments.camera_observation:
        from camera_observation.live import run_camera_live
        return run_camera_live(arguments, sys.modules[__name__])
    log_launch_configuration(arguments)
    run_live(
        arguments.model_path,
        arguments.scenario,
        arguments.seed,
        arguments.policy_seed,
        arguments.deterministic,
        arguments.max_steps,
        arguments.frame_delay,
        arguments.result_hold_seconds,
        arguments.radar_range_m,
        arguments.guided_missiles,
        arguments.snapshot_dir,
        not arguments.no_missile_vfx,
        arguments.visual_demo,
        arguments.demo_separation_m,
        arguments.diagnostics_json,
        DEFAULT_MISSILE_LAUNCH_CEILING_M,
        arguments.combat_presentation,
        arguments.playback_rate,
        arguments.terminal_hold_seconds,
        reseed_policy_after_load=True,
    )


if __name__ == "__main__":
    main()
