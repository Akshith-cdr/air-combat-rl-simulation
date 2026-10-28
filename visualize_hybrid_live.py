"""Onscreen Panda3D demonstration of the final hybrid PPO on REC-003."""

import argparse
import os
import secrets
import time
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
from live_visualization import PreserveTerminalVisuals

loadPrcFileData("hybrid_live", "win-size 1280 720\nshow-frame-rate-meter 1\n")


DEFAULT_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
)
DEFAULT_SCENARIO = "REC-003"


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


def make_live_environment(stats_path: Path):
    vector_env = DummyVecEnv(
        [lambda: PreserveTerminalVisuals(FighterJetEnv(hybrid_fire=True))]
    )
    env = VecNormalize.load(str(stats_path), vector_env)
    env.action_space = vector_env.action_space
    env.training = False
    env.norm_reward = False
    return env


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
):
    hud.setText(
        "POLICY: Hybrid PPO\n"
        f"MODE: {action_mode}\n"
        f"SCENARIO: {scenario_id} / randomized\n"
        f"STEP: {step}   TIME: {simulation_time:6.2f} s\n"
        f"RANGE: {float(raw_observation[10]):8.2f} m\n"
        f"cos(ATA): {float(raw_observation[11]):+.5f}\n"
        f"FIRE: {fire_text}\n"
        f"MISSILE: {missile_text}\n"
        f"TERMINAL: {terminal_reason or 'RUNNING'}"
    )


def hold_terminal_frame(sim, seconds: float, frame_delay: float):
    if sim.missile_vfx is not None:
        sim.missile_vfx.hold(seconds, frame_delay)
        return
    deadline = time.monotonic() + max(0.0, seconds)
    refresh_delay = frame_delay if frame_delay > 0.0 else 1.0 / 60.0
    while time.monotonic() < deadline:
        sim.graphicsEngine.renderFrame()
        time.sleep(refresh_delay)


def run_live(
    model_path: Path,
    scenario_id: str,
    simulation_seed: int | None,
    policy_seed: int,
    deterministic: bool,
    max_steps: int,
    frame_delay: float,
    result_hold_seconds: float,
    vfx: bool = True,
    cinematic_camera: bool = True,
    missile_slowmo: float = 8.0,
):
    stats_path = normalization_path(model_path)
    missing = [str(path) for path in (model_path, stats_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing live-visualization input: " + ", ".join(missing))
    scenario = scenario_by_id(scenario_id)
    seed = resolve_simulation_seed(simulation_seed)
    print(f"Simulation seed: {seed}")

    np.random.seed(policy_seed)
    th.manual_seed(policy_seed)
    env = make_live_environment(stats_path)
    hud = None
    event_hud = None
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        if not isinstance(model.policy, HybridActorCriticPolicy):
            raise TypeError("The supplied checkpoint is not a hybrid PPO policy")
        if model.observation_space.shape != (19,) or model.action_space.shape != (5,):
            raise ValueError("Live hybrid visualization requires 19 observations and 5 actions")
        if tuple(model.policy.action_net.weight.shape) != (4, 64):
            raise ValueError("Expected four continuous Gaussian action means")
        if tuple(model.policy.log_std.shape) != (4,):
            raise ValueError("Expected four continuous Gaussian log standard deviations")
        if tuple(model.policy.fire_net.weight.shape) != (1, 64):
            raise ValueError("Expected one Bernoulli fire logit")

        reset_scenario = asdict(scenario)
        reset_scenario["seed"] = int(seed)
        env.set_options({"enemy_challenge_scenario": reset_scenario})
        env.seed(seed)
        observation = env.reset()
        raw_env = env.venv.envs[0].unwrapped
        sim = raw_env.sim
        sim.hud_text.hide()
        if vfx:
            sim.enable_missile_effects(cinematic=cinematic_camera, slowmo=missile_slowmo)
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

        mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
        simulation_time = 0.0
        launched = False
        terminal_reason = ""
        last_fire_text = "0 — NO REQUEST"
        last_missile_text = "READY" if raw_env._missile_available() else "UNAVAILABLE"
        print(
            f"Hybrid live visualization: model={model_path}, vecnormalize={stats_path}, "
            f"scenario={scenario.scenario_id}, simulation_seed={seed}, "
            f"policy_seed={policy_seed}, mode={mode.lower()}"
        )

        for step in range(1, max_steps + 1):
            if sim.missile_vfx is not None:
                sim.missile_vfx.before_step()
            action, _ = model.predict(observation, deterministic=deterministic)
            action_values = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_values.shape != (5,):
                raise ValueError(f"Unexpected hybrid action shape: {action_values.shape}")
            if action_values[4] not in (0.0, 1.0):
                raise ValueError(f"Hybrid fire output is not binary: {action_values[4]}")

            # This is the only actuation path. FighterJetEnv validates the gate and
            # calls JetSim.fire_missile(), which owns the real projectile and physics.
            observation, _, done, infos = env.step(action)
            info = infos[0]
            weapon_info = info.get("weapon", {})
            simulation_time += float(info.get("dt", 0.0))
            launched |= bool(weapon_info.get("valid_fire_attempt", False))
            last_fire_text = fire_state(int(action_values[4]), weapon_info)
            last_missile_text = missile_state(raw_env, weapon_info, launched)
            terminal_reason = (
                info.get("terminal_reason", "") or "environment_terminated"
                if bool(done[0])
                else ""
            )
            raw_observation = env.get_original_obs()[0].copy()

            if weapon_info.get("valid_fire_attempt", False):
                event_hud.setText("REAL MISSILE LAUNCHED")
                print(
                    f"Real missile launch: step={step}, t={simulation_time:.3f}s, "
                    f"range={raw_observation[10]:.3f}m, cosATA={raw_observation[11]:.6f}, "
                    f"active_projectiles={len(sim.active_missiles)}"
                )
            if weapon_info.get("missile_hit_this_step", False):
                event_hud.setText("MISSILE HIT TARGET\nREAL SIMULATOR HIT EVENT")
                print(
                    f"Real missile hit: step={step}, t={simulation_time:.3f}s, "
                    f"hit_count={sim.missile_hit_count}"
                )

            update_hud(
                hud,
                scenario.scenario_id,
                mode,
                step,
                simulation_time,
                raw_observation,
                last_fire_text,
                last_missile_text,
                terminal_reason,
            )
            if sim.missile_vfx is not None:
                sim.missile_vfx.render_step(float(info.get("dt", 0.0)), frame_delay)
            else:
                sim.graphicsEngine.renderFrame()
                if frame_delay > 0.0:
                    time.sleep(frame_delay)

            if bool(done[0]):
                print(f"Live episode ended at step {step}: {terminal_reason}")
                hold_terminal_frame(sim, result_hold_seconds, frame_delay)
                return terminal_reason

        terminal_reason = "visualization_step_limit"
        update_hud(
            hud,
            scenario.scenario_id,
            mode,
            max_steps,
            simulation_time,
            env.get_original_obs()[0].copy(),
            last_fire_text,
            last_missile_text,
            terminal_reason,
        )
        print(f"Live episode reached the {max_steps}-step safety limit")
        hold_terminal_frame(sim, result_hold_seconds, frame_delay)
        return terminal_reason
    finally:
        if hud is not None:
            hud.destroy()
        if event_hud is not None:
            event_hud.destroy()
        env.venv.envs[0].unwrapped.sim.destroy()
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--seed",
        type=int,
        help="Simulation seed (default: a fresh system-random 32-bit seed).",
    )
    parser.add_argument("--policy-seed", type=int, default=41003)
    parser.add_argument("--max-steps", type=int, default=MAX_EVALUATION_STEPS)
    parser.add_argument("--frame-delay", type=float, default=1.0 / 60.0)
    parser.add_argument("--result-hold-seconds", type=float, default=3.0)
    parser.add_argument("--no-vfx", action="store_true", help="Disable missile effects.")
    parser.add_argument("--no-cinematic-camera", action="store_true",
                        help="Keep the aircraft chase camera during launch and impact.")
    parser.add_argument("--missile-slowmo", type=float, default=8.0,
                        help="Presentation speed factor during actual missile flight (default: 8).")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use Gaussian means and Bernoulli mode (default: stochastic sampling).",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_live(
        arguments.model_path,
        arguments.scenario,
        arguments.seed,
        arguments.policy_seed,
        arguments.deterministic,
        arguments.max_steps,
        arguments.frame_delay,
        arguments.result_hold_seconds,
        not arguments.no_vfx,
        not arguments.no_cinematic_camera,
        arguments.missile_slowmo,
    )
