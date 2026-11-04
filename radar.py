"""Read-only, north-up radar projection and Panda3D HUD rendering.

World X maps right and world Y maps up. Altitude is deliberately omitted.
Physics and policy observations never consume radar output.
"""

from dataclasses import dataclass
import math

from projectile_motion import vector3
from live_navigation_panel import to_map


@dataclass(frozen=True)
class RadarConfig:
    range_m: float = 10000.0
    center: tuple[float, float, float] | None = None  # None follows own aircraft.
    boundary: str = "clamp"

    def __post_init__(self):
        if not math.isfinite(self.range_m) or self.range_m <= 0:
            raise ValueError("radar range must be finite and positive")
        if self.center is not None:
            vector3(self.center)
        if self.boundary not in ("clamp", "hide"):
            raise ValueError("radar boundary must be clamp or hide")


@dataclass(frozen=True)
class RadarContact:
    entity_id: str
    team: str
    kind: str
    position: tuple[float, float, float]


def project_contacts(contacts, own_position, config=RadarConfig()):
    """Map identities to (contact, x, y, clipped), in the closed unit disc.

    Invalid coordinates are omitted. Contacts exactly on range remain visible.
    A new mapping each call ensures reset/destruction cannot leave stale blips.
    """
    center = vector3(config.center if config.center is not None else own_position)
    result = {}
    for contact in contacts:
        try:
            position = vector3(contact.position)
        except (ValueError, TypeError, OverflowError):
            continue
        mapped = to_map(position, center, config.range_m)
        if mapped is None:
            continue
        x, y, clipped = mapped
        if clipped and config.boundary == "hide":
            continue
        if contact.entity_id in result:
            raise ValueError("duplicate radar entity identity")
        result[contact.entity_id] = (contact, x, y, clipped)
    return result


class PandaRadar:
    """One marker per entity; no limit per team or projectile type."""

    def __init__(self, parent):
        from panda3d.core import LineSegs, TextNode
        self.root = parent.attachNewNode("combat_radar")
        # Parent is ShowBase.a2dBottomRight, which follows window resizing.
        # Radius .27 plus .07 margin from the bottom and right edges.
        self.root.setPos(-0.34, 0, 0.34)
        self.root.setScale(0.27)
        self.root.setBin("fixed", 25)
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        lines = LineSegs("radar_grid")
        lines.setColor(0.3, 0.6, 0.6, 1)
        for i in range(65):
            angle = i * math.tau / 64
            point = (math.sin(angle), 0, math.cos(angle))
            if i == 0:
                lines.moveTo(*point)
            else:
                lines.drawTo(*point)
        for a, b in [((-1, 0, 0), (1, 0, 0)), ((0, 0, -1), (0, 0, 1))]:
            lines.moveTo(*a)
            lines.drawTo(*b)
        self.root.attachNewNode(lines.create())
        label = TextNode("radar_label")
        label.setAlign(TextNode.ACenter)
        label.setText("RADAR  +Y UP\nGREEN FRIEND / RED FOE")
        self.label = self.root.attachNewNode(label)
        self.label.setScale(.11)
        self.label.setZ(1.25)
        self.markers = {}

    def update(self, projected, own_team="friendly"):
        from panda3d.core import TextNode
        for identity in self.markers.keys() - projected.keys():
            self.markers.pop(identity).removeNode()
        for identity, (contact, x, y, clipped) in projected.items():
            if identity not in self.markers:
                text = TextNode(identity)
                text.setAlign(TextNode.ACenter)
                text.setText("+" if contact.kind == "aircraft" else "o")
                self.markers[identity] = self.root.attachNewNode(text)
                self.markers[identity].setScale(.17 if contact.kind == "aircraft" else .12)
            marker = self.markers[identity]
            marker.setPos(x, -.01, y)
            color = (0.2, 1, 0.45, 1) if contact.team == own_team else (1, .25, .2, 1)
            marker.setColor(*color[:3], .55 if clipped else 1)

    def reset(self):
        for marker in self.markers.values():
            marker.removeNode()
        self.markers.clear()

    def destroy(self):
        self.reset()
        self.root.removeNode()
