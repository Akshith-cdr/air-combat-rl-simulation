"""Read-only numerical telemetry shared by the viewer and speed diagnostic."""

import math


def heading_rate(previous, current, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Heading rate requires positive simulation time")
    return ((current - previous + 180) % 360 - 180) / dt


def combat_telemetry(sim, previous_heading, dt, simulation_time, playback_rate, elapsed):
    player_speed = float(sim.player.velocity.length())
    enemy_speed = float(sim.enemy.velocity.length())
    return {
        "player_speed_m_s": player_speed, "player_speed_km_h": player_speed * 3.6,
        "enemy_speed_m_s": enemy_speed, "enemy_speed_km_h": enemy_speed * 3.6,
        "distance_m": float((sim.enemy.position - sim.player.position).length()),
        "player_heading_deg": float(sim.player.yaw),
        "player_heading_rate_deg_s": heading_rate(previous_heading, sim.player.yaw, dt),
        "simulation_time_s": simulation_time, "requested_playback_rate": playback_rate,
        "achieved_playback_rate": simulation_time / max(1e-9, elapsed),
    }
