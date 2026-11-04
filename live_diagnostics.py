"""Opt-in wall-clock diagnostics; no file I/O in simulation/render callbacks."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import hashlib


class FlightDiagnostics:
    def __init__(self):
        self.step = 0
        self.phase = "episode"
        self.samples = {}
        self.flights = []
        self._states = []
        self._previous_frame = None
        self.metadata = {}
        self.decisions = []
        self.transitions = []
        self.frames = []
        self._fire_context = None
        self.firing_authority = "learned_policy"
        self.speed_telemetry = []

    def record_prediction(self, observation, action, raw_observation):
        self.decisions.append({"step": self.step,
                               "normalized_observation": observation.reshape(-1).tolist(),
                               "raw_observation": raw_observation.reshape(-1).tolist(),
                               "predicted_action": action.reshape(-1).tolist()})

    @staticmethod
    def aircraft_state(actor):
        return {"position": tuple(actor.position), "velocity": tuple(actor.velocity),
                "hpr": (float(actor.yaw), float(actor.pitch), float(actor.roll)),
                "controls": tuple(float(getattr(actor, name)) for name in
                                  ("pitch_input", "roll_input", "yaw_input", "speed_input"))}

    def observe_environment(self, env):
        """Observe the actual raw env input and setter, independently of predict()."""
        self.firing_authority = getattr(env, "firing_authority", "learned_policy")
        original_setter = env._set_player_controls
        applied = {}

        def setter(action):
            original_setter(action)
            applied["clipped"] = action.reshape(-1).tolist()
            applied["controls"] = self.aircraft_state(env.sim.player)["controls"]

        env._set_player_controls = setter
        original_fire = env._process_hybrid_fire

        def authorized_fire(*args, **kwargs):
            self._fire_context = ("scripted_demo_physical_gate" if self.firing_authority == "scripted_demonstration"
                                  else "environment_hybrid_fire")
            try:
                return original_fire(*args, **kwargs)
            finally:
                self._fire_context = None

        env._process_hybrid_fire = authorized_fire
        original_step = env.step

        def observed_step(action):
            decision = self.decisions[-1] if self.decisions else None
            before = self.aircraft_state(env.sim.player)
            received = action.reshape(-1).tolist()
            result = original_step(action)
            matched = (decision is not None and decision["step"] == self.step
                       and received == decision["predicted_action"])
            low, high = env.action_space.low, env.action_space.high
            expected = [max(float(lo), min(float(hi), float(value)))
                        for value, lo, hi in zip(received, low, high)]
            controls_matched = (applied.get("clipped") == expected
                                and list(applied.get("controls", ())) == expected[:4]
                                and list(self.aircraft_state(env.sim.player)["controls"]) == expected[:4])
            self.transitions.append({"step": self.step, "received_action": received,
                                     "clipped_action": applied.get("clipped"),
                                     "applied_controls": applied.get("controls"),
                                     "prediction_matches_input": matched,
                                     "controls_match_clipped_input": controls_matched,
                                     "player_before": before,
                                     "player_after": self.aircraft_state(env.sim.player),
                                     "enemy_after": self.aircraft_state(env.sim.enemy),
                                     "enemy_rng_sha256": hashlib.sha256(repr(env.sim.rng.getstate()).encode()).hexdigest(),
                                     "observation_after": result[0].tolist(),
                                     "reward": float(result[1]), "weapon": result[4]["weapon"]})
            if self.firing_authority == "scripted_demonstration":
                self.transitions[-1]["demo_firing"] = dict(env.last_demo_fire)
            if not matched or not controls_matched:
                raise RuntimeError("RL audit: policy output did not match the applied environment action")
            return result

        env.step = observed_step

    def record_frame(self, sim, frame_buffer, alpha):
        """Observe submitted scene transforms before they are restored."""
        player = sim.player.root
        rotation = player.getQuat(sim.render)
        expected_camera = (player.getPos(sim.render) + rotation.getForward() * sim.camera_offset.y
                           + rotation.getUp() * sim.camera_offset.z)
        row = {"step": self.step, "alpha": alpha,
               "player_scene": tuple(player.getPos(sim.render)),
               "player_numerical": tuple(sim.player.position),
               "camera_position": tuple(sim.camera.getPos(sim.render)),
               "camera_target": sim.camera_target,
               "camera_error_m": math.dist(tuple(expected_camera), tuple(sim.camera.getPos(sim.render))),
               "missiles": []}
        simple = getattr(sim, "simple_missile_visuals", None)
        if simple is not None:
            for state in sim.active_missiles:
                identity = state["missile_id"]
                node = simple.visuals.get(identity)
                if node is None:
                    row["missiles"].append({"id": identity, "missing_visual": True})
                    continue
                position = tuple(state["position"])
                previous = simple.previous.get(identity, position)
                expected = tuple(p * (1-alpha) + q * alpha for p, q in zip(previous, position))
                visual_position = tuple(node.getPos(sim.render))
                direction = node.getQuat(sim.render).getForward()
                velocity = state["velocity"]
                contact = sim.radar_contacts.get(identity)
                row["missiles"].append({"id": identity, "numerical": position,
                                        "visual": visual_position,
                                        "interpolation_error_m": math.dist(expected, visual_position),
                                        "velocity_alignment": direction.dot(velocity.normalized())
                                        if velocity.lengthSquared() else 1.0,
                                        "radar_error_m": math.dist(position, contact[0].position) if contact else None})
        self.frames.append(row)

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(name, []).append({
                "step": self.step, "phase": self.phase,
                "milliseconds": (time.perf_counter() - start) * 1000})
            if name == "render_frame":
                now = time.perf_counter()
                if self._previous_frame is not None:
                    self.samples.setdefault("frame_interval", []).append({
                        "step": self.step, "phase": self.phase,
                        "milliseconds": (now - self._previous_frame) * 1000})
                self._previous_frame = now

    def attach(self, sim):
        sim.live_diagnostics = self
        for name, label in (("update_missiles", "missile_integration"),
                            ("update_radar", "radar_update")):
            original = getattr(sim, name)

            def timed(*args, _original=original, _label=label, **kwargs):
                with self.measure(_label):
                    return _original(*args, **kwargs)

            setattr(sim, name, timed)
        launch = sim.launch_guided_missile

        def tracked_launch(*args, **kwargs):
            with self.measure("launch"):
                state = launch(*args, **kwargs)
            if state is not None:
                target = sim.enemy if state["target_id"] == "enemy" else sim.player
                position = tuple(state["guided"].position)
                record = {"missile_id": state["missile_id"], "launch_step": self.step,
                          "authorization": self._fire_context or "direct_launch_outside_fire_gate",
                          "launch_position": position,
                          "launch_target_position": tuple(target.position),
                          "initial_separation_m": math.dist(position, tuple(target.position)),
                          "updates": []}
                self.flights.append(record)
                self._states.append((state, record))
            return state

        sim.launch_guided_missile = tracked_launch

    def record_step(self):
        for state, record in self._states:
            if "terminal" in record:
                continue
            missile = state["guided"]
            record["updates"].append({"step": self.step, "age": missile.age,
                                      "position": missile.position,
                                      "velocity": missile.velocity})
            event = missile.terminal
            if event is not None:
                record["terminal"] = {
                    "reason": event.reason, "step": self.step,
                    "duration_seconds": event.age, "position": event.position,
                    "target_position": event.target_position,
                    "path_distance_m": missile.distance,
                    "displacement_m": math.dist(record["launch_position"], event.position)
                    if event.position is not None else None,
                    "terminal_separation_m": math.dist(event.position, event.target_position)
                    if event.position is not None and event.target_position is not None else None,
                    "closest_approach_m": event.closest_approach}

    def write(self, path):
        summaries = {}
        for name, rows in self.samples.items():
            values = sorted(row["milliseconds"] for row in rows)
            summaries[name] = {"count": len(values), "mean_ms": sum(values) / len(values),
                               "p95_ms": values[min(len(values) - 1, int(.95 * len(values)))],
                               "max_ms": max(values)}
        audit = {"predict_calls": len(self.decisions), "environment_steps": len(self.transitions),
                 "all_predictions_applied": bool(self.transitions) and all(
                     row["prediction_matches_input"] and row["controls_match_clipped_input"]
                     for row in self.transitions),
                 "unique_actions": len({tuple(row["predicted_action"]) for row in self.decisions}),
                 "unique_normalized_observations": len({tuple(row["normalized_observation"]) for row in self.decisions}),
                 "unique_rendered_player_positions": len({tuple(row["player_scene"]) for row in self.frames}),
                 "player_path_m": sum(math.dist(row["player_before"]["position"], row["player_after"]["position"])
                                      for row in self.transitions),
                 "all_launches_environment_authorized": all(
                     row["authorization"] in ("environment_hybrid_fire", "scripted_demo_physical_gate")
                     for row in self.flights)
                     if self.flights else None}
        if self.firing_authority == "scripted_demonstration":
            audit.update(firing_authority=self.firing_authority, learned_firing_capability=False,
                         policy_action_vectors_received_unchanged=audit["all_predictions_applied"])
        payload = {"metadata": self.metadata, "audit": audit,
                                         "summary": summaries, "samples": self.samples,
                                         "decisions": self.decisions, "transitions": self.transitions,
                                         "frames": self.frames, "flights": self.flights}
        if self.speed_telemetry:
            payload["speed_telemetry"] = self.speed_telemetry
        # Scripted demonstrations must never overwrite an existing diagnostic.
        with Path(path).open("x" if self.firing_authority == "scripted_demonstration" else "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        return summaries
