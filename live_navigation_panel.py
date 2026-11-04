"""Own-aircraft-only Panda3D overlay; no policy, simulator or weapon imports.

Aircraft-centred world XY map: +X right, +Y up, equal metre scale, Z altitude.
Range is the circle radius; off-map supplied navigation points retain bearing at
the rim. Forward is projected from the aircraft's real world-space orientation.
Call update immediately before rendering with the existing simulation timestamp.
Waypoints are optional explicit stationary navigation data, never inferred from
other actors. This display neither steps the simulation nor modifies its state.
"""
from collections import deque
import math


def finite_vector(value, length=3):
    try:
        result = tuple(float(x) for x in value)
        return result if len(result) == length and all(math.isfinite(x) for x in result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def to_map(point, origin, range_m):
    """Return normalized XY and off-range flag, or None for invalid input."""
    point, origin = finite_vector(point), finite_vector(origin)
    if point is None or origin is None or not math.isfinite(range_m) or range_m <= 0:
        return None
    x, y = (point[0] - origin[0]) / range_m, (point[1] - origin[1]) / range_m
    distance = math.hypot(x, y)
    if not math.isfinite(distance):
        return None
    return (x / distance, y / distance, True) if distance > 1 else (x, y, False)


def forward_xy(value):
    vector = finite_vector(value)
    if vector is None:
        return None
    length = math.hypot(vector[0], vector[1])
    return (vector[0] / length, vector[1] / length) if length > 1e-8 else None


class OwnNavigationPanel:
    """Text-free bottom-right overlay; ten simulated seconds, <=240 samples."""
    def __init__(self, parent, range_m=2000.0):
        from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                                 GeomVertexFormat, GeomVertexWriter, LineSegs,
                                 TransparencyAttrib)

        if not math.isfinite(range_m) or range_m <= 0:
            raise ValueError("Navigation range must be finite and positive")
        self.range_m = float(range_m)
        self.radius, self.cx, self.cz = .185, -.225, .225
        self.root = parent.attachNewNode("own-navigation-panel")
        self.root.setBin("fixed", 30)
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        # A filled disk, not a rectangular card. Aspect2D keeps the circle round.
        vertices = GeomVertexData("navigation-disk", GeomVertexFormat.getV3(), Geom.UHStatic)
        writer = GeomVertexWriter(vertices, "vertex")
        writer.addData3(self.cx, 0, self.cz)
        for index in range(65):
            angle = index * math.tau / 64
            writer.addData3(self.cx + self.radius * math.cos(angle), 0,
                            self.cz + self.radius * math.sin(angle))
        triangles = GeomTriangles(Geom.UHStatic)
        for index in range(64):
            triangles.addVertices(0, index + 1, index + 2)
        geometry = Geom(vertices)
        geometry.addPrimitive(triangles)
        disk = GeomNode("navigation-background")
        disk.addGeom(geometry)
        background = self.root.attachNewNode(disk)
        background.setColor(.015, .035, .055, .78)
        background.setTransparency(TransparencyAttrib.MAlpha)
        rings = LineSegs("navigation-rings")
        rings.setColor(.25, .48, .60, .55)
        for fraction in (.5, 1.):
            for index in range(65):
                angle = index * math.tau / 64
                xyz = (self.cx + self.radius * fraction * math.cos(angle), 0,
                       self.cz + self.radius * fraction * math.sin(angle))
                (rings.moveTo if index == 0 else rings.drawTo)(*xyz)
        rings.moveTo(self.cx - self.radius, 0, self.cz)
        rings.drawTo(self.cx + self.radius, 0, self.cz)
        rings.moveTo(self.cx, 0, self.cz - self.radius)
        rings.drawTo(self.cx, 0, self.cz + self.radius)
        ring_node = self.root.attachNewNode(rings.create())
        ring_node.setTransparency(TransparencyAttrib.MAlpha)
        ring_node.setBin("fixed", 31)
        self.dynamic = self.root.attachNewNode("navigation-dynamic")
        self.dynamic.setBin("fixed", 32)
        self.trail = deque(maxlen=240)
        self.last_time = None

    def _clear_dynamic(self):
        self.dynamic.removeNode()
        self.dynamic = self.root.attachNewNode("navigation-dynamic")
        self.dynamic.setBin("fixed", 32)

    def _vertex(self, x, y):
        return self.cx + x * self.radius, 0, self.cz + y * self.radius

    def update(self, position, forward, simulation_time, waypoints=None,
               reached=(), current_index=None):
        """Read-only snapshot. Optional reached indices must come from caller state."""
        from panda3d.core import LineSegs

        self._clear_dynamic()
        position, heading = finite_vector(position), forward_xy(forward)
        try:
            timestamp = float(simulation_time)
        except (TypeError, ValueError, OverflowError):
            timestamp = float("nan")
        if position is None or not math.isfinite(timestamp) or timestamp < 0:
            self.trail.clear()
            self.last_time = None
            return
        if self.last_time is not None and timestamp < self.last_time:
            self.trail.clear()  # A reset starts a new trail, never links episodes.
        if timestamp != self.last_time:
            self.trail.append((timestamp, position))
        self.last_time = timestamp
        while self.trail and timestamp - self.trail[0][0] > 10.:
            self.trail.popleft()
        lines = LineSegs("own-navigation-state")
        lines.setThickness(2)
        lines.setColor(.25, .85, 1, 1)
        drawing = False
        for _, point in self.trail:
            mapped = to_map(point, position, self.range_m)
            if mapped is None or mapped[2]:
                drawing = False  # No false connecting segment across off-map gaps.
                continue
            (lines.drawTo if drawing else lines.moveTo)(*self._vertex(*mapped[:2]))
            drawing = True
        lines.setColor(.20, .50, 1, 1)
        if heading is not None:
            x, y = heading
            for i, (a, b) in enumerate(((.12*x, .12*y), (-.07*x-.07*y, -.07*y+.07*x),
                                        (-.07*x+.07*y, -.07*y-.07*x), (.12*x, .12*y))):
                (lines.moveTo if i == 0 else lines.drawTo)(*self._vertex(a, b))
        else:
            lines.moveTo(*self._vertex(-.04, 0))
            lines.drawTo(*self._vertex(.04, 0))
        if waypoints is not None:
            try:
                points = list(waypoints)
                reached = set(reached)
                if len(points) > 3:
                    raise ValueError("At most three stationary navigation points")
                for index, point in enumerate(points):
                    mapped = to_map(point, position, self.range_m)
                    if mapped is None:
                        continue
                    x, y, edge = mapped
                    color = (0.3, 1, .5, 1) if index in reached else (1, .8, .2, 1)
                    lines.setColor(*color)
                    lines.moveTo(*self._vertex(x-.025, y))
                    lines.drawTo(*self._vertex(x+.025, y))
                    lines.moveTo(*self._vertex(x, y-.025))
                    lines.drawTo(*self._vertex(x, y+.025))
            except (TypeError, ValueError):
                pass  # No fabricated position or status when optional data is absent.
        self.dynamic.attachNewNode(lines.create())

    def destroy(self):
        self.root.removeNode()


def update_own_panel(panel, aircraft, simulation_time):
    """Adapter only for own aircraft; no access to any other simulation entity."""
    try:
        position = aircraft.position
        forward = aircraft.get_forward_vector()
    except (AttributeError, TypeError, ValueError):
        position, forward = None, None
    panel.update(position, forward, simulation_time)
