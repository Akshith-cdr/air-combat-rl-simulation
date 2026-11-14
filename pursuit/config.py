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

# --- Collision --------------------------------------------------------------
# The simulator has no collision model. An episode in which our aircraft comes
# within this 3D distance of the target is ended and counted as a failure.
COLLISION_RADIUS_M = 10.0

# --- Reward (Eq. 2 of the paper is written from this block) -----------------
CLOSURE_SPEED_SCALE = 250.0     # m/s of band-distance reduction that gives r_closure = 1
GEOMETRY_RANGE_SCALE = 500.0    # m outside the band over which the potential falls to 0
GEOMETRY_PROGRESS_SCALE = 0.10
W_CLOSURE = 1.0
W_ALIGNMENT = 1.0
W_GEOMETRY = 1.0
W_SPEED = 0.5
W_ALTITUDE = 0.5
W_AOA = 1.0
W_BOUNDARY = 0.5               # NOT USED: PursuitEnv has no boundary shaping term (kept so
                               # recorded configs of trained runs stay comparable)
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
