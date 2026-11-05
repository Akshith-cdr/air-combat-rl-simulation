"""Camera-only mode in the existing viewer, scene, radar and chase camera."""

from dataclasses import fields
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from .pipeline import write_json
from .simulation import ROOT, FrozenFlightController, OriginalFlightSession, digest
from .task import CameraConfig, CameraState


def load_camera_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    if manifest.get('schema') != 'camera-observation-v1' or results.get('status') != 'trained_and_evaluated':
        raise ValueError('Expected a completed, separate camera-observation run')
    for name, expected in {**manifest['protected_sha256'], **manifest['source_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Camera training source/asset compatibility mismatch: {name}')
    if digest(directory/'camera_policy.zip') != results['camera_policy_sha256']:
        raise ValueError('Camera policy hash mismatch')
    config = CameraConfig(**{f.name: manifest['camera_config'][f.name] for f in fields(CameraConfig)})
    policy = PPO.load(str(directory/'camera_policy.zip'), device='cpu')
    if policy.observation_space.shape != (10,) or getattr(policy.action_space, 'n', None) != 2:
        raise ValueError('Camera checkpoint must have 10 camera observations and 2 camera actions')
    policy.policy.set_training_mode(False)
    return policy, config, manifest


def run_camera_live(arguments, viewer):
    torch.set_num_threads(1)
    policy, config, manifest = load_camera_run(arguments.camera_run)
    flight = FrozenFlightController()
    if flight.hashes != manifest['flight_checkpoint_sha256']:
        raise ValueError('Camera run requires its original frozen flight checkpoint/statistics')
    session = OriginalFlightSession(flight, config, headless=False)
    hud = event_hud = None
    try:
        viewer.configure_live_view(session.env.sim, arguments.radar_range_m, visible_missiles=False)
        sim = session.env.sim
        frame = session.reset(arguments.seed)
        sim.hud_text.hide()
        hud = viewer.OnscreenText(text='', pos=(-1.29, .91), scale=.041,
                                 align=viewer.TextNode.ALeft, fg=(1, 1, 1, 1),
                                 shadow=(0, 0, 0, .9), mayChange=True)
        event_hud = viewer.OnscreenText(text='', pos=(0, .70), scale=.085,
                                       align=viewer.TextNode.ACenter, fg=(1, .86, .08, 1),
                                       shadow=(0, 0, 0, .95), mayChange=True)
        state = CameraState(config)
        buffer = viewer.PhysicsFrameBuffer(sim)
        schedule = viewer.PlaybackSchedule(arguments.playback_rate)
        positions, physics_steps = [], 0

        def update_display():
            hud.setText(
                'CAMERA OBSERVATION / LEARNED SIGHTING SCHEDULE\n'
                'FLIGHT: FROZEN PPO / CAMERA ACTION: RECORD OR WAIT\n'
                f'PLAYER: {sim.player.current_speed:.1f} m/s / {sim.player.current_speed*3.6:.0f} km/h\n'
                f'ENEMY: {sim.enemy.current_speed:.1f} m/s / {sim.enemy.current_speed*3.6:.0f} km/h\n'
                f'HEADING: {sim.player.yaw:.1f} deg / ALTITUDE: {sim.player.altitude:.0f} m\n'
                f'SIM: {physics_steps*session.env.simulation_dt:.2f} s / PLAYBACK: {arguments.playback_rate:g}x\n'
                f'SIGHTINGS: {sum(r["recorded"] for r in state.rows)} / EXPOSURES LEFT: {state.remaining}\n'
                f'TRACK: {"FRESH SIGHTING" if state.age < config.freshness_seconds else "NO FRESH SIGHTING"}\n'
                f'SEED: {arguments.seed} / CAMERA POLICY: DETERMINISTIC')

        def before(_sim):
            buffer.before_step()

        def after(_sim):
            nonlocal physics_steps
            physics_steps += 1
            positions.append(tuple(sim.player.position))
            buffer.after_step()
            update_display()
            count = schedule.frames(session.env.simulation_dt, arguments.frame_delay)
            for sample in range(1, count+1):
                viewer.render_aircraft_frame(sim, buffer, sample/count)
                if arguments.frame_delay:
                    schedule.pace(session.env.simulation_dt, count)

        update_display()
        viewer.render_aircraft_frame(sim)
        if arguments.snapshot_dir:
            viewer.save_snapshot(sim, arguments.snapshot_dir, 'camera-initial')
        photographed = False
        for _ in range(min(arguments.max_steps, config.horizon)):
            observation = state.observation(frame)
            action = int(policy.predict(observation, deterministic=True)[0])
            _, row = state.record(action, frame)
            if row['recorded']:
                event_hud.setText('SIGHTING RECORDED')
            elif action:
                event_hud.setText('NO USABLE CAMERA SIGHTING')
            frame = session.advance(before_step=before, after_step=after)
            if row['recorded'] and arguments.snapshot_dir and not photographed:
                viewer.save_snapshot(sim, arguments.snapshot_dir, 'camera-sighting')
                photographed = True
            if session.reason != 'horizon':
                break
        reason = (session.reason if session.reason != 'horizon' or state.step_index == config.horizon
                  else 'viewer_limit')
        event_hud.setText('CAMERA SESSION COMPLETE')
        viewer.render_aircraft_frame(sim)
        if arguments.snapshot_dir:
            viewer.save_snapshot(sim, arguments.snapshot_dir, 'camera-terminal')
        flight.verify()
        result = dict(task='non-weapon camera observation', seed=arguments.seed,
                      action_authority='learned camera policy; frozen PPO flight',
                      camera_run=str(arguments.camera_run.resolve()), weapon_requests=0, launches=0,
                      physics_steps=physics_steps, rows=state.rows, player_positions=positions,
                      radar_contact_ids=sorted(sim.radar_contacts), metrics=state.metrics(reason),
                      frozen_flight_hashes=flight.hashes, frozen_flight_unchanged=True)
        if arguments.diagnostics_json:
            arguments.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
            if arguments.diagnostics_json.exists():
                raise FileExistsError('Camera diagnostics require a new output path')
            write_json(arguments.diagnostics_json, result)
        viewer.hold_terminal_frame(sim, arguments.terminal_hold_seconds, arguments.frame_delay)
        print(json.dumps(result['metrics'], indent=2))
        return result
    finally:
        if event_hud:
            event_hud.destroy()
        if hud:
            hud.destroy()
        session.close()
