"""Launch authorization shared by the environment and original simulator."""

import math


MIN_MISSILE_LAUNCH_DISTANCE_M = 500.0
# Existing long-range viewer ceiling; reward geometry retains its 100 m reference.
DEFAULT_MISSILE_LAUNCH_CEILING_M = 750.0


def launch_distance_allowed(distance_m):
    """The inclusive minimum applies to finite 3D aircraft-to-target range."""
    return math.isfinite(distance_m) and distance_m >= MIN_MISSILE_LAUNCH_DISTANCE_M
