"""Explicit viewer-only firing authority. Never imported by training/gameplay."""

import math

from fighter_jet_env import FighterJetEnv, FIRE_COS_ATA_GATE
from weapon_config import MIN_MISSILE_LAUNCH_DISTANCE_M, DEFAULT_MISSILE_LAUNCH_CEILING_M


DEMO_LABEL = "PPO FLIGHT CONTROL / FIRE-CONTROL SYSTEM"
DEMO_WARNING = "FIRE-CONTROL ENVELOPE: 500-600 m, cos(ATA) > 0.95"
DEFAULT_REQUEST_CEILING_M = 600.0
MIN_REQUEST_RANGE_M = MIN_MISSILE_LAUNCH_DISTANCE_M
PREFERRED_MIN_RANGE_M = MIN_MISSILE_LAUNCH_DISTANCE_M


def validate_demo_options(enabled, distance, guided_missiles,
                          launch_range_m=DEFAULT_MISSILE_LAUNCH_CEILING_M):
    if distance is not None and not enabled:
        raise ValueError("--demo-launch-distance-m requires --scripted-demo-firing")
    if not enabled:
        return None
    distance = DEFAULT_REQUEST_CEILING_M if distance is None else distance
    if not math.isfinite(distance) or not 550 <= distance <= 600:
        raise ValueError("Demo request ceiling must be finite and within 550..600 m inclusive; old 400 m scripted commands are unsupported")
    if not guided_missiles:
        raise ValueError("Scripted demo firing requires --guided-missiles")
    if launch_range_m != DEFAULT_MISSILE_LAUNCH_CEILING_M:
        raise ValueError("Scripted demo firing conflicts with legacy launch-range settings")
    return float(distance)


class ScriptedDemoFlightEnv(FighterJetEnv):
    """Receive PPO actions unchanged; choose firing authority at the fire hook.

    This changes only the viewer's request source and explicit range ceiling.
    The parent step, controls, gate, weapon initialization and physics are used
    without modification. Reward numbers here are demonstration outcomes only.
    """

    firing_authority = "scripted_demonstration"

    def __init__(self, *, demo_launch_distance_m=DEFAULT_REQUEST_CEILING_M, missile_guidance=False,
                 hybrid_fire=True, **kwargs):
        distance = validate_demo_options(True, demo_launch_distance_m, missile_guidance)
        if not hybrid_fire or "launch_range_gate" in kwargs or "fire_range_gate" in kwargs:
            raise ValueError("Demo uses the hybrid fire hook and its own explicit launch ceiling")
        self.last_demo_fire = None
        super().__init__(hybrid_fire=True, missile_guidance=missile_guidance,
                         launch_range_gate=distance, **kwargs)

    def reset(self, **kwargs):
        self.last_demo_fire = None
        return super().reset(**kwargs)

    def _process_hybrid_fire(self, fire_bit, current_range, current_cos_ata):
        # These are the parent's current numerical geometry, not PPO observations.
        # The action vector and learned fire bit are never overwritten.
        current_range = self._distance_to_target()
        if not math.isfinite(current_range):
            window = "invalid_range"
        elif current_range < MIN_REQUEST_RANGE_M:
            window = "below_minimum"
        elif current_range >= self.launch_range_gate:
            window = "at_or_above_ceiling"
        else:
            window = "preferred"
        request = window == "preferred"
        # Diagnostic snapshot only. Authorization remains exclusively in the
        # parent implementation below; capture ammunition BEFORE it is consumed.
        safety = {
            "target_alive": not getattr(self.sim, "target_destroyed", False),
            "simulation_active": not self.sim.game_over,
            "missile_available": bool(self._missile_available()),
            "range_at_or_above_minimum": bool(current_range >= MIN_REQUEST_RANGE_M),
            "range_below_ceiling": bool(current_range < self.launch_range_gate),
            "alignment_valid": bool(current_cos_ata > FIRE_COS_ATA_GATE),
        }
        before = len(self.sim.active_missiles)
        result = super()._process_hybrid_fire(float(request), current_range, current_cos_ata)
        rejected = [name for name, passed in safety.items() if not passed] if request and result[2] else []
        if request and result[2] and not rejected:
            rejected = ["launch_not_created"]
        self.last_demo_fire = {
            "firing_authority": self.firing_authority,
            "original_ppo_fire_bit": float(fire_bit),
            "original_ppo_fire_request": float(fire_bit) >= .5,
            "scripted_fire_request": request,
            "engagement_window": window,
            "request_suppression_reason": None if request else window,
            "safety_before_authorization": safety,
            "parent_fire_result": list(map(bool, result)),
            "physical_rejection_reasons": rejected,
            "parent_gate_rejected": bool(result[2] and rejected != ["launch_not_created"]),
            "accepted_launch": bool(result[1]),
            "physical_gate_result": "accepted" if result[1] else "rejected" if result[2] else "not_requested",
            "physical_gate_rejected": bool(result[2]),
            "actual_missile_launch": len(self.sim.active_missiles) > before,
            "range_m": float(current_range), "cos_ata": float(current_cos_ata),
            "launch_ceiling_m": self.launch_range_gate,
            "minimum_request_range_m": MIN_REQUEST_RANGE_M,
            "preferred_minimum_range_m": PREFERRED_MIN_RANGE_M,
        }
        return result
