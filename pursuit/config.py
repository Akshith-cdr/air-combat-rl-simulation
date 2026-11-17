"""Every constant that defines the task. Training and evaluation read only this.

Keeping one source means the launch gate used in training is, by construction,
the gate used in evaluation (reviewer comment M1).
"""

from weapon_config import MAX_MISSILE_LAUNCH_DISTANCE_M, MIN_MISSILE_LAUNCH_DISTANCE_M

# --- Simulation -----------------------------------------------------------
SIMULATION_DT = 0.05            # s per environment step
MAX_EPISODE_STEPS = 2000        # 100 s of simulated time

# --- Launch gate (identical in training and evaluation) --------------------
BAND_MIN_M = float(MIN_MISSILE_LAUNCH_DISTANCE_M)   # 500 m, inclusive
BAND_MAX_M = float(MAX_MISSILE_LAUNCH_DISTANCE_M)   # 1000 m, inclusive
COS_ATA_GATE = 0.95             # aircraft velocity within ~18.2 deg of the line of sight

# --- Target ------------------------------------------------------------------
# The legacy patrol compared a clockwise desired heading with the counter-clockwise
# yaw, so the target wandered instead of flying to its waypoints. True = fixed.
TARGET_PATROL_CORRECT_HEADING = True

# --- Collision --------------------------------------------------------------
# The simulator has no collision model. An episode in which our aircraft comes
# within this 3D distance of the target is ended and counted as a failure.
COLLISION_RADIUS_M = 10.0

# --- Reward (Eq. 3 of the paper is written from this block) -----------------
# Progress shaping Phi(s') - Phi(s) on every step, including the last (env.potential).
# Over an episode it sums to Phi(s_T) - Phi(s_0): net progress towards an aligned
# position in the band. Along any closed loop of states it sums to exactly zero, so
# circling cannot earn reward (the earlier clipped terms could be farmed that way).
# Pilots: gamma*Phi(s') - Phi(s) with Phi(s_T) = 0 and small weights (A = 20, K = 4)
# gave too weak a signal; untrained policies learned to leave the boundary instead.
SHAPING_RANGE_WEIGHT = 100.0    # potential gained by moving from >= 5 km outside the band into it
SHAPING_RANGE_SCALE_M = 5000.0
SHAPING_ALIGN_WEIGHT = 20.0     # potential for pointing at the target near the band
SHAPING_NEAR_SCALE_M = 500.0    # metres outside the band over which the alignment term fades
# After launch only: a small per-step penalty for flying closer than SEPARATION_RADIUS_M
# to the target, -W_SEPARATION * clip((R - d) / R, 0, 1). Always <= 0, so it cannot be
# farmed. (In the reviewed version every PPO collision happened after launch.)
SEPARATION_RADIUS_M = 300.0
W_SEPARATION = 1.0
W_SPEED = 0.5
W_ALTITUDE = 0.5
W_AOA = 1.0
TIME_PENALTY = 0.01
LAUNCH_REWARD = 5.0
HIT_REWARD = 100.0
BOUNDARY_PENALTY = -25.0
COLLISION_PENALTY = -50.0
MIN_SAFE_SPEED = 150.0
MIN_SAFE_ALTITUDE = 250.0
MAX_SAFE_AOA_DEG = 15.0


def as_dict():
    """All task constants, for saving next to every checkpoint and result."""
    return {name: value for name, value in globals().items()
            if name.isupper() and isinstance(value, (int, float))}
