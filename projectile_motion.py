"""Deterministic projectile motion, optional PN guidance and swept geometry.

Positions share a Cartesian frame; velocity is distance per second.
Guidance is a numerical point-mass model, independent of rendering and damage.
"""

import math
from dataclasses import dataclass, field, replace
from copy import copy


Vector = tuple[float, float, float]


def vector3(value) -> Vector:
    """Validate a world-frame vector (metres, m/s, or m/s²)."""
    result = tuple(float(x) for x in value)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError("expected three finite vector components")
    return result


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _scale(a, scale):
    return tuple(x * scale for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.hypot(*a)


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


@dataclass(frozen=True)
class GuidanceConfig:
    """Arcade point-mass parameters, SI units; angles are in degrees.

    These are simulation tuning values, not a model of a real weapon.
    Seeker loss is latched: coast without guidance, or terminate immediately.
    """

    navigation_gain: float = 3.0
    boost_acceleration: float = 120.0
    boost_seconds: float = 3.0
    coast_deceleration: float = 0.0
    max_speed: float = 650.0
    max_lateral_acceleration: float = 250.0
    max_turn_rate_deg: float = 120.0
    lifetime: float = 20.0
    max_distance: float = 25000.0
    proximity_radius: float = 50.0
    floor: float = -1000.0
    integration_dt: float = 0.01
    seeker_range: float | None = None
    seeker_half_angle_deg: float | None = None
    lock_loss: str = "coast"

    def __post_init__(self):
        for name in ("navigation_gain", "boost_acceleration", "boost_seconds",
                     "coast_deceleration", "max_speed", "max_lateral_acceleration",
                     "max_turn_rate_deg", "lifetime", "max_distance",
                     "proximity_radius", "integration_dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if min(self.max_speed, self.lifetime, self.max_distance, self.integration_dt) <= 0:
            raise ValueError("speed, lifetime, distance and integration_dt must be positive")
        if not math.isfinite(self.floor):
            raise ValueError("floor must be finite")
        if self.seeker_range is not None and (
                not math.isfinite(self.seeker_range) or self.seeker_range <= 0):
            raise ValueError("seeker_range must be finite and positive")
        if self.seeker_half_angle_deg is not None and (
                not math.isfinite(self.seeker_half_angle_deg)
                or not 0 <= self.seeker_half_angle_deg <= 180):
            raise ValueError("seeker_half_angle_deg must be in 0..180")
        if self.lock_loss not in ("coast", "terminate"):
            raise ValueError("lock_loss must be coast or terminate")


@dataclass(frozen=True)
class TargetTrack:
    entity_id: str
    team: str
    position: Vector
    velocity: Vector
    alive: bool = True


@dataclass(frozen=True)
class MissileEvent:
    missile_id: str
    team: str
    target_id: str
    reason: str
    age: float
    position: Vector | None
    target_position: Vector | None
    closest_approach: float
    step_fraction: float


@dataclass
class GuidedMissile:
    missile_id: str
    team: str
    target_id: str
    position: Vector
    velocity: Vector
    config: GuidanceConfig = field(default_factory=GuidanceConfig)
    age: float = 0.0
    distance: float = 0.0
    closest_approach: float = math.inf
    locked: bool = True
    terminal: MissileEvent | None = None
    lateral_acceleration: Vector = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.position = vector3(self.position)
        self.velocity = vector3(self.velocity)
        if not all((self.missile_id, self.team, self.target_id)):
            raise ValueError("missile, team and target identities are required")
        if _norm(self.velocity) > self.config.max_speed:
            raise ValueError("initial velocity exceeds configured max_speed")


def closest_approach(start, end, other_start, other_end):
    """Exact distance and fraction for two swept, linear world-frame segments."""
    relative = _sub(start, other_start)
    delta = _sub(_sub(end, other_end), relative)
    squared = _dot(delta, delta)
    fraction = max(0.0, min(1.0, -_dot(relative, delta) / squared)) if squared else 0.0
    return _norm(_add(relative, _scale(delta, fraction))), fraction


def guidance_acceleration(position, velocity, target: TargetTrack,
                          config: GuidanceConfig) -> Vector:
    """Velocity-normal PN, with bounded LOS acquisition for opening/rear targets.

    r = target - missile, omega = (r × relative_velocity) / |r|².
    Closing, forward targets use N * closing_speed * (LOS omega × v_hat).
    Positive X is right, positive Y forward, positive Z up in the simulator.
    """
    relative = _sub(target.position, position)
    relative_velocity = _sub(target.velocity, velocity)
    distance, speed = _norm(relative), _norm(velocity)
    if distance < 1e-9 or speed < 1e-9:
        return (0.0, 0.0, 0.0)
    omega = _scale(_cross(relative, relative_velocity), 1.0 / distance**2)
    closing = -_dot(relative, relative_velocity) / distance
    direction = _scale(velocity, 1.0 / speed)
    los = _scale(relative, 1.0 / distance)
    alignment = _dot(los, direction)
    if closing <= 0 or alignment <= 0:
        # PN assumes a closing engagement. Acquire/recover a target outside that
        # geometry using velocity-normal LOS pursuit, under the same limits.
        normal = _sub(los, _scale(direction, alignment))
        if _norm(normal) < 1e-9 and alignment < 0:
            # Exact antiparallel LOS has no preferred turn plane. Choose one
            # deterministically; a vertical launch needs a different reference.
            reference = (0.0, 0.0, 1.0) if abs(direction[2]) < .9 else (1.0, 0.0, 0.0)
            normal = _cross(direction, reference)
            normal = _scale(normal, 1.0 / _norm(normal))
        command = _scale(normal, config.navigation_gain * speed**2 / distance)
    else:
        command = _scale(_cross(omega, direction), config.navigation_gain * closing)
    magnitude = _norm(command)
    limit = min(config.max_lateral_acceleration,
                speed * math.radians(config.max_turn_rate_deg))
    return _scale(command, min(1.0, limit / magnitude)) if magnitude else command


def integrate_guided_velocity(velocity, acceleration, age, dt, config):
    """Rotate at bounded angular rate without artificial steering speed gain."""
    speed = _norm(velocity)
    direction = _scale(velocity, 1.0 / speed) if speed > 1e-9 else (0.0, 1.0, 0.0)
    magnitude = _norm(acceleration)
    if magnitude and speed > 1e-9:
        angle = min(magnitude * dt / speed, math.radians(config.max_turn_rate_deg) * dt)
        direction = _add(_scale(direction, math.cos(angle)),
                         _scale(acceleration, math.sin(angle) / magnitude))
    thrust = config.boost_acceleration if age < config.boost_seconds else -config.coast_deceleration
    return _scale(direction, max(0.0, min(config.max_speed, speed + thrust * dt)))


def advance_guided_missile(missile: GuidedMissile, target: TargetTrack,
                           target_previous_position, dt: float) -> MissileEvent | None:
    """Advance numerical state and resolve proximity/expiry; never apply damage.

    Target motion is interpolated over its actual caller-supplied segment.
    Substeps split exactly at burnout and expiration. Events are latched so
    repeated calls cannot generate duplicate hits. VFX and scene nodes are absent.
    """
    dt = validate_dt(dt)
    if missile.terminal is not None:
        return None
    start_age = missile.age
    config = missile.config

    def finish(reason, target_position=None):
        try:
            position = vector3(missile.position)
        except (ValueError, TypeError, OverflowError):
            position = None
        missile.terminal = MissileEvent(
            missile.missile_id, missile.team, missile.target_id, reason,
            missile.age, position, target_position, missile.closest_approach,
            (missile.age - start_age) / dt if dt else 0.0)
        return missile.terminal

    try:
        vector3(missile.position)
        vector3(missile.velocity)
        vector3(target.position)
        vector3(target.velocity)
        previous = vector3(target_previous_position)
        if not all(math.isfinite(x) and x >= 0 for x in (missile.age, missile.distance)):
            raise ValueError("invalid missile clock/distance")
    except (ValueError, TypeError, OverflowError):
        return finish("invalid_state")
    if target.entity_id != missile.target_id or target.team == missile.team:
        return finish("invalid_target")
    missile.closest_approach = min(missile.closest_approach,
                                   _norm(_sub(missile.position, previous)))
    if not target.alive:
        return finish("target_destroyed", target.position)

    elapsed = 0.0
    while True:
        fraction = elapsed / dt if dt else 0.0
        target_start = _add(previous, _scale(_sub(target.position, previous), fraction))
        if missile.age >= config.lifetime - 1e-12:
            return finish("expired", target_start)
        if missile.distance >= config.max_distance - 1e-9:
            return finish("range", target_start)
        if missile.position[2] <= config.floor:
            return finish("floor", target_start)
        relative = _sub(target_start, missile.position)
        distance, speed = _norm(relative), _norm(missile.velocity)
        if distance > 1e-9:
            outside_range = config.seeker_range is not None and distance > config.seeker_range
            outside_angle = (config.seeker_half_angle_deg is not None and speed > 1e-9
                             and _dot(relative, missile.velocity) / (distance * speed)
                             < math.cos(math.radians(config.seeker_half_angle_deg)))
            missile.locked = missile.locked and not (outside_range or outside_angle)
        if not missile.locked and config.lock_loss == "terminate":
            return finish("seeker_lost", target_start)
        h = min(config.integration_dt, dt - elapsed, config.lifetime - missile.age)
        if missile.age < config.boost_seconds:
            h = min(h, config.boost_seconds - missile.age)
        track = TargetTrack(target.entity_id, target.team, target_start, target.velocity)
        acceleration = (guidance_acceleration(missile.position, missile.velocity, track, config)
                        if missile.locked else (0.0, 0.0, 0.0))
        missile.lateral_acceleration = acceleration
        velocity = integrate_guided_velocity(missile.velocity, acceleration, missile.age, h, config)
        segment_velocity = _scale(_add(missile.velocity, velocity), 0.5)
        end = linear_position(missile.position, segment_velocity, h)
        travel = _norm(_sub(end, missile.position))
        limit = min(1.0, (config.max_distance - missile.distance) / travel) if travel else 1.0
        reason = "range" if limit < 1.0 else None
        if end[2] <= config.floor and end[2] < missile.position[2]:
            floor_limit = (config.floor - missile.position[2]) / (end[2] - missile.position[2])
            if floor_limit <= limit:
                limit, reason = floor_limit, "floor"
        h *= limit
        end = linear_position(missile.position, segment_velocity, h)
        target_end = _add(previous, _scale(_sub(target.position, previous),
                                          (elapsed + h) / dt if dt else 0.0))
        contact = swept_contact_fraction(missile.position, end, target_start,
                                         target_end, config.proximity_radius)
        if contact is not None:
            h *= contact
            end = linear_position(missile.position, segment_velocity, h)
            target_end = _add(target_start, _scale(_sub(target_end, target_start), contact))
            reason = "hit"
        closest, _ = closest_approach(missile.position, end, target_start, target_end)
        missile.closest_approach = min(missile.closest_approach, closest)
        missile.distance += _norm(_sub(end, missile.position))
        missile.position = end
        missile.velocity = integrate_guided_velocity(
            missile.velocity, acceleration, missile.age, h, config)
        missile.age += h
        elapsed += h
        if reason:
            return finish(reason, target_end)
        if missile.distance >= config.max_distance - 1e-9:
            return finish("range", target_end)
        if missile.age >= config.lifetime - 1e-12:
            return finish("expired", target_end)
        if elapsed >= dt - 1e-12:
            return None


def advance_guided_salvo(missiles, tracks, previous_positions, dt):
    """Resolve a guided salvo chronologically, independently of list order.

    Predict each numerical path before applying damage. The earliest hit per
    target wins (identity breaks exact-time ties). Other missiles stop at that
    target's destruction time; an earlier expiration still takes precedence.
    Return new terminal events sorted within the caller's timestep.
    """
    dt = validate_dt(dt)
    candidates, winners = {}, {}
    for missile in missiles:
        if missile.missile_id in candidates:
            raise ValueError("duplicate missile identity")
        candidate = copy(missile)
        event = advance_guided_missile(candidate, tracks[missile.target_id],
                                       previous_positions[missile.target_id], dt)
        candidates[missile.missile_id] = (candidate, event)
        if event is not None and event.reason == "hit":
            key = (event.step_fraction, event.missile_id)
            prior = winners.get(missile.target_id)
            if prior is None or key < (prior.step_fraction, prior.missile_id):
                winners[missile.target_id] = event
    events = []
    for missile in missiles:
        candidate, event = candidates[missile.missile_id]
        winner = winners.get(missile.target_id)
        if (winner is not None and winner.missile_id != missile.missile_id
                and (event is None or event.step_fraction >= winner.step_fraction)):
            cutoff = dt * winner.step_fraction
            track = tracks[missile.target_id]
            previous = previous_positions[missile.target_id]
            target_at_cutoff = _add(previous, _scale(_sub(track.position, previous),
                                                    winner.step_fraction))
            candidate = copy(missile)
            event = advance_guided_missile(candidate, replace(track, position=target_at_cutoff),
                                           previous, cutoff)
            if event is None:
                event = MissileEvent(candidate.missile_id, candidate.team, candidate.target_id,
                                     "target_destroyed", candidate.age, candidate.position,
                                     target_at_cutoff, candidate.closest_approach, winner.step_fraction)
            else:
                event = replace(event, reason="target_destroyed", step_fraction=winner.step_fraction)
            candidate.terminal = event
        missile.__dict__.update(candidate.__dict__)
        if event is not None:
            events.append(event)
    return sorted(events, key=lambda event: (event.step_fraction, event.missile_id))


def validate_dt(dt):
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    return dt


def linear_position(position, velocity, dt):
    dt = validate_dt(dt)
    return tuple(float(p) + float(v) * dt for p, v in zip(position, velocity))


def sphere_crossings(start, end, center, radius):
    """Return entry/exit fractions of an infinite line through a sphere.

    Fractions refer to start + fraction * (end - start). None means no
    crossing, including a stationary point; callers handle initial overlap.
    """
    offset = tuple(float(p) - float(c) for p, c in zip(start, center))
    delta = tuple(float(q) - float(p) for p, q in zip(start, end))
    a = sum(v * v for v in delta)
    if a == 0:
        return None
    b = sum(p * v for p, v in zip(offset, delta))
    c = sum(p * p for p in offset) - radius * radius
    discriminant = b * b - a * c
    if discriminant < 0:
        return None
    # Stable quadratic roots, including tangency at the line origin.
    q = -b - math.copysign(math.sqrt(discriminant), b)
    if q == 0:
        return (0.0, 0.0)
    return tuple(sorted((q / a, c / q)))


def swept_contact_fraction(start, end, other_start, other_end, radius):
    """Earliest contact in [0, 1] for two linearly moving points and a radius."""
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and non-negative")
    relative_start = tuple(float(p) - float(q) for p, q in zip(start, other_start))
    relative_end = tuple(float(p) - float(q) for p, q in zip(end, other_end))
    if sum(p * p for p in relative_start) <= radius * radius:
        return 0.0
    roots = sphere_crossings(relative_start, relative_end, (0, 0, 0), radius)
    if roots is not None and 0.0 <= roots[0] <= 1.0:
        return roots[0]
    return None
