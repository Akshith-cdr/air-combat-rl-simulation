"""Initial conditions: the training sampler and the evaluation suites.

Conventions (used everywhere in this package and in the paper)
--------------------------------------------------------------
* compass heading: degrees clockwise from world +Y ("north"), in [0, 360).
* bearing: direction to the target measured clockwise from our aircraft's
  nose, in (-180, 180]. 0 = dead ahead, +90 = right, 180 = behind.
* target relative heading: target compass heading minus ours. 0 = the target
  flies the same way we do, 180 = it flies the opposite way.

Panda3D stores yaw counter-clockwise, so an aircraft with compass heading psi
has yaw = -psi. Both aircraft are given a velocity and a nose that agree.
"""

import math
import random
from dataclasses import asdict, dataclass

import numpy as np

# The six geometries of the reviewed draft (REC-001..REC-006, evaluate_ppo.py).
# Our aircraft starts at (0, 0, 1000) m, compass heading 0, 150 m/s.
# Fields: id, target world position (x, y, z), target speed, target compass heading.
REVIEWED_SIX = (
    ("REC-001", (-1200.0, 2200.0, 700.0), 140.0, 0.0),
    ("REC-002", (-800.0, 2600.0, 900.0), 155.0, 30.0),
    ("REC-003", (-400.0, 3000.0, 1100.0), 170.0, 60.0),
    ("REC-004", (0.0, 3400.0, 1300.0), 185.0, 90.0),
    ("REC-005", (400.0, 3800.0, 1500.0), 200.0, 120.0),
    ("REC-006", (800.0, 4200.0, 1700.0), 215.0, 150.0),
)

SWEEP_BEARINGS = tuple(range(0, 360, 30))           # 12 sectors
SWEEP_TARGET_HEADINGS = (0, 90, 180, 270)           # relative to ours
SWEEP_RANGES = (1500.0, 3000.0, 4500.0)
SWEEP_REPEATS = 7                                   # 12*4*3*7 = 1008 episodes
SWEEP_ALTITUDE = 1500.0
SWEEP_PLAYER_SPEED = 200.0
SWEEP_TARGET_SPEED = 190.0


def wrap180(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int                       # simulator RNG seed (target waypoint choice)
    player_heading: float           # compass deg
    player_speed: float             # m/s
    altitude: float                 # our altitude, m
    range_m: float                  # initial horizontal range, m
    bearing: float                  # deg, clockwise from our nose
    delta_altitude: float           # target altitude minus ours, m
    target_relative_heading: float  # deg
    target_speed: float             # m/s

    @property
    def target_heading(self):
        return (self.player_heading + self.target_relative_heading) % 360.0

    @property
    def bearing_sector(self):
        """30 deg sector centred on a multiple of 30: 0, 30, ..., 330."""
        return int(round((self.bearing % 360.0) / 30.0)) % 12 * 30

    def as_dict(self):
        return {**asdict(self), "target_heading": self.target_heading,
                "bearing_sector": self.bearing_sector}


def training_scenario(rng: np.random.Generator, episode: int) -> Scenario:
    """Training distribution: target anywhere around us, any heading."""
    return Scenario(
        scenario_id=f"TRAIN-{episode:07d}",
        seed=int(rng.integers(0, 2**32)),
        player_heading=float(rng.uniform(0.0, 360.0)),
        player_speed=float(rng.uniform(160.0, 260.0)),
        altitude=float(rng.uniform(1000.0, 2200.0)),
        range_m=float(rng.uniform(1500.0, 5000.0)),
        bearing=wrap180(rng.uniform(-180.0, 180.0)),
        delta_altitude=float(rng.uniform(-500.0, 500.0)),
        target_relative_heading=float(rng.uniform(0.0, 360.0)),
        target_speed=float(rng.uniform(140.0, 240.0)),
    )


def episode_seeds(base_seed, count):
    """Unique uint32 seeds; copied from evaluate_visualize_hybrid_combat.py."""
    seed = int(np.random.SeedSequence(base_seed, spawn_key=(0,)).generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(seed)
    result, used = [], set()
    while len(result) < count:
        candidate = int(rng.integers(0, 2**32, dtype=np.uint32))
        if candidate not in used:
            used.add(candidate)
            result.append(candidate)
    return result


def reviewed_six_scenario(name, position, speed, heading, seed):
    x, y, z = position
    return Scenario(scenario_id=name, seed=int(seed), player_heading=0.0, player_speed=150.0,
                    altitude=1000.0, range_m=math.hypot(x, y),
                    bearing=wrap180(math.degrees(math.atan2(x, y))), delta_altitude=z - 1000.0,
                    target_relative_heading=float(heading), target_speed=float(speed))


def reviewed_six_suite(count=1000, base_seed=987654321):
    """The reviewed draft's 1,000-episode protocol: seed -> one of six geometries.

    Episode seeds and the seed -> geometry choice are generated exactly as in
    evaluate_visualize_hybrid_combat.py, so episode k uses the same geometry
    and simulator seed as before.
    """
    scenarios = []
    for seed in episode_seeds(base_seed, count):
        name, position, speed, heading = random.Random(seed).choice(REVIEWED_SIX)
        scenarios.append(reviewed_six_scenario(name, position, speed, heading, seed))
    return scenarios


def sweep_suite(base_seed=20261114, repeats=SWEEP_REPEATS, deltas=(0.0,), prefix="SW",
                headings=SWEEP_TARGET_HEADINGS, ranges=SWEEP_RANGES):
    """All-round test: 12 bearings x target headings x ranges x altitude offsets x repeats.

    Repeats differ only in the simulator seed, i.e. in the target's random
    waypoints. Seeds come from a fixed generator independent of training.
    `deltas` are target-minus-own altitude offsets (m).
    """
    rng = np.random.default_rng(base_seed)
    scenarios = []
    for bearing in SWEEP_BEARINGS:
        for rel in headings:
            for distance in ranges:
                for dz in deltas:
                    for k in range(repeats):
                        alt = "" if len(deltas) == 1 else f"-z{int(dz):+d}"
                        scenarios.append(Scenario(
                            scenario_id=f"{prefix}-b{bearing:03d}-h{rel:03d}-r{int(distance)}{alt}-{k}",
                            seed=int(rng.integers(0, 2**32)), player_heading=0.0,
                            player_speed=SWEEP_PLAYER_SPEED, altitude=SWEEP_ALTITUDE,
                            range_m=distance, bearing=wrap180(bearing), delta_altitude=float(dz),
                            target_relative_heading=float(rel), target_speed=SWEEP_TARGET_SPEED))
    return scenarios


# Altitude offsets of the second test sweep (training samples -500..+500 m).
SWEEP_ALT_DELTAS = (-450.0, -300.0, -150.0, 150.0, 300.0, 450.0)


def sweep_alt_suite():
    """The sweep's geometries with the target 150-450 m above or below us
    (12 x 4 x 3 x 6 = 864 episodes, one seed each, seeds independent of the sweep)."""
    return sweep_suite(base_seed=20261116, repeats=1, deltas=SWEEP_ALT_DELTAS, prefix="SA")


def development_suite():
    """Hard subset flown during training for the learning curves (72 episodes).

    The all-round 48-episode set of the reviewed version saturated at 100% early.
    This one has only the cells that failed most often for the three baselines on
    a separate exploration set (sweep geometry, seed 4242, not a test suite):
    head-on or crossing targets (relative heading 90, 180, 270) starting 1500 or
    3000 m away, from all 12 bearings, with the target 450 m below us.
    Its own seeds; never a test set.
    """
    return sweep_suite(base_seed=777002, repeats=1, deltas=(-450.0,), prefix="DEV",
                       headings=(90, 180, 270), ranges=(1500.0, 3000.0))


def tuning_suite():
    """Baseline-gain tuning set (12 x 4 x 2 x 3 = 288 episodes): the sweep's bearings
    and target headings at 1500 and 3000 m, with the target 300 m below, level with,
    or 300 m above us. Seeds differ from every test and dev suite."""
    return sweep_suite(base_seed=31337002, repeats=1, deltas=TUNE_DELTAS, prefix="TU",
                       ranges=(1500.0, 3000.0))


TUNE_DELTAS = (-300.0, 0.0, 300.0)


SUITES = {"reviewed_six": reviewed_six_suite, "sweep": sweep_suite, "sweep_alt": sweep_alt_suite,
          "dev": development_suite, "tune": tuning_suite}


def suite(name):
    return SUITES[name]()
