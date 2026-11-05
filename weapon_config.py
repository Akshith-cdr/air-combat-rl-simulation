"""Missile launch envelope shared by the environment and the simulator."""

import math


# The fire action is accepted only while the 3D aircraft-to-target range lies
# inside [MIN, MAX] metres, both ends inclusive. Requests outside the band are
# rejected by the environment and by the simulator's launcher.
MIN_MISSILE_LAUNCH_DISTANCE_M = 500.0
MAX_MISSILE_LAUNCH_DISTANCE_M = 1000.0
# Default environment launch ceiling: the same 1000 m limit.
DEFAULT_MISSILE_LAUNCH_CEILING_M = MAX_MISSILE_LAUNCH_DISTANCE_M


def launch_distance_allowed(distance_m):
    """True when the finite 3D range lies within the inclusive launch band."""
    return (math.isfinite(distance_m)
            and distance_m >= MIN_MISSILE_LAUNCH_DISTANCE_M
            and distance_m <= MAX_MISSILE_LAUNCH_DISTANCE_M)
