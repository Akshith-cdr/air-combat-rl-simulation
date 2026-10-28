"""Panda3D missile presentation driven exclusively by real simulator events.

Textures are generated in memory. No asset downloads, particle tasks, global
random calls, or physics changes are required. Live playback interpolates visual
nodes between actual physics samples; simulation state is never interpolated.
"""

from dataclasses import dataclass
import math
import random
import time

from panda3d.core import (
    CardMaker, ColorBlendAttrib, NodePath, PNMImage, PointLight, Texture,
    TransparencyAttrib, Vec3, Geom, GeomNode, GeomTriangles,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter,
)


@dataclass
class Particle:
    node: object
    position: Vec3
    velocity: Vec3
    age: float
    lifetime: float
    size: float
    growth: float
    color: tuple
    kind: str
    spin: float


def _noise(u, v, frequency):
    """Smooth deterministic value noise, independent of every runtime RNG."""
    x, y = u * frequency + 17, v * frequency + 23
    ix, iy = math.floor(x), math.floor(y)
    sx, sy = x - ix, y - iy
    sx, sy = sx * sx * (3 - 2 * sx), sy * sy * (3 - 2 * sy)

    def corner(a, b):
        value = math.sin(a * 127.1 + b * 311.7) * 43758.5453
        return value - math.floor(value)

    lower = corner(ix, iy) * (1 - sx) + corner(ix + 1, iy) * sx
    upper = corner(ix, iy + 1) * (1 - sx) + corner(ix + 1, iy + 1) * sx
    return lower * (1 - sy) + upper * sy


def _texture(kind, size=128):
    """Soft radial textures with coherent turbulence and transparent edges."""
    image = PNMImage(size, size, 4)
    for y in range(size):
        for x in range(size):
            u = (x + 0.5) / size * 2.0 - 1.0
            v = (y + 0.5) / size * 2.0 - 1.0
            radius = math.hypot(u, v)
            turbulence = ((_noise(u, v, 4) * 0.6 + _noise(u, v, 9) * 0.3
                           + _noise(u, v, 19) * 0.1) - 0.5) * 2
            edge = max(0.0, 1.0 - radius)
            if kind == "glow":
                alpha = max(0.0, math.exp(-radius * radius * 6) - math.exp(-6))
                rgb = (1.0, 1.0, 1.0)
            elif kind == "ring":
                alpha = math.exp(-((radius - 0.78) / 0.035) ** 2) * edge * 4
                rgb = (1.0, 0.85, 0.56)
            elif kind == "fire":
                edge = max(0.0, edge + turbulence * 0.18 * min(1, edge * 8))
                alpha = min(1.0, edge * 4) * max(0.0, 0.7 + turbulence * 0.5)
                heat = min(1.0, edge * 1.6 + turbulence * 0.5)
                rgb = (1.0, 0.18 + heat * 0.77, 0.025 + heat ** 3 * 0.65)
            else:
                alpha = min(1.0, edge * 3) * (0.55 + turbulence * 0.22)
                shade = 0.75 + turbulence * 0.2
                rgb = (shade, shade, shade)
            image.setXelA(x, y, *rgb, min(1.0, max(0.0, alpha)))
    texture = Texture("missile_" + kind)
    texture.load(image)
    texture.setMinfilter(Texture.FTLinear)
    texture.setMagfilter(Texture.FTLinear)
    texture.setWrapU(Texture.WMClamp)
    texture.setWrapV(Texture.WMClamp)
    return texture


class MissileVFX:
    """Bounded, explicitly stepped VFX with a separate presentation clock."""

    BLAST_DURATION = 4.0
    MAX_PARTICLES = 420

    def __init__(self, sim, cinematic=True, slowmo=8.0):
        self.sim = sim
        self.cinematic = bool(cinematic)
        self.slowmo = max(1.0, float(slowmo))
        self.rng = random.Random(73019)  # Never consume the enemy/policy RNG.
        self.root = sim.render.attachNewNode("missile_vfx")
        self.root.setLightOff(1)
        self.root.setShaderOff(1)
        self.root.setDepthWrite(False)
        self.textures = {kind: _texture(kind) for kind in ("glow", "fire", "smoke", "ring")}
        card = CardMaker("vfx_sprite")
        card.setFrame(-1, 1, -1, 1)
        self.card = card.generate()
        self.debris_mesh = self._debris_mesh()
        self.particles = []
        self.flight = None
        self.impact = None
        self.impact_age = 0.0
        self.blast_light = None
        self.shockwave = None
        self.flash = None
        self.actor_samples = None
        self.rendered_flight_frames = 0
        self.launch_count = 0
        self.blast_count = 0

    def _sprite(self, kind, position, size, color, additive=False):
        # Share geometry, but give every particle its own transform/render state.
        node = NodePath(self.card).copyTo(self.root)
        node.setName("vfx_" + kind)
        node.setTexture(self.textures[kind], 1)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setTwoSided(True)
        node.setBillboardPointEye()
        node.setPos(position)
        node.setScale(size)
        node.setColor(*color)
        if additive:
            node.setAttrib(ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ))
        return node

    def _particle(self, kind, position, velocity, lifetime, size, growth, color):
        if len(self.particles) >= self.MAX_PARTICLES:
            self.particles.pop(0).node.removeNode()
        if kind == "debris":
            node = NodePath(self.debris_mesh).copyTo(self.root)
            node.setPos(position)
            node.setScale(size)
            node.setColor(*color)
            node.setDepthWrite(True)
            node.setTwoSided(True)
        else:
            node = self._sprite(kind, position, size, color, kind in ("fire", "glow"))
        self.particles.append(Particle(
            node, Vec3(position), Vec3(velocity), 0.0, lifetime, size, growth,
            color, kind, self.rng.uniform(-70, 70),
        ))

    @staticmethod
    def _debris_mesh():
        data = GeomVertexData("metal_fragment", GeomVertexFormat.getV3(), Geom.UHStatic)
        writer = GeomVertexWriter(data, "vertex")
        for point in ((-1, -0.3, 0), (0.8, -0.3, -0.2), (0, 1, 0.15), (0, 0, 0.7)):
            writer.addData3(*point)
        triangles = GeomTriangles(Geom.UHStatic)
        for face in ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)):
            triangles.addVertices(*face)
            triangles.closePrimitive()
        geom = Geom(data)
        geom.addPrimitive(triangles)
        node = GeomNode("blast_debris")
        node.addGeom(geom)
        return node

    def _light(self, name, position, color, attenuation):
        light = PointLight(name)
        light.setColor(color)
        light.setAttenuation(attenuation)
        node = self.sim.render.attachNewNode(light)
        node.setPos(position)
        # Only the two aircraft need these lights; don't relight 250 clouds.
        self.sim.player.visual_root.setLight(node)
        self.sim.enemy.visual_root.setLight(node)
        return node

    def _remove_light(self, node):
        if node is not None and not node.isEmpty():
            self.sim.player.visual_root.clearLight(node)
            self.sim.enemy.visual_root.clearLight(node)
            node.removeNode()

    def on_launch(self, state):
        """Keep the physics projectile and use its copy for smooth rendering."""
        if self.flight is not None:
            self._clear_flight()
        body = state["node"].copyTo(self.root)
        body.setName("missile_presentation_body")
        body.show()
        body.setLightOff(1)
        body.setShaderOff(1)
        body.setColorScale(1.35, 1.35, 1.35, 1)
        body.setDepthWrite(True)
        state["node"].hide()
        direction = Vec3(state["velocity"])
        direction.normalize()
        position = Vec3(state["spawn_pos"])
        flames = [self._sprite("glow", position, 1, (1, 0.48, 0.08, 1), True)
                  for _ in range(7)]
        self.flight = {
            "state": state, "body": body, "previous": position,
            "current": Vec3(position), "direction": direction, "flames": flames,
            "light": self._light("missile_engine", position, (2.4, 0.8, 0.16, 1),
                                 (1, 0.04, 0.014)),
            "ended": False, "hit": False, "age": 0.0, "trail_distance": 0.0,
            "launch_pending": True,
        }
        self.launch_count += 1

    def on_motion(self, state):
        if self.flight is not None and self.flight["state"] is state:
            self.flight["current"] = Vec3(state["node"].getPos(self.sim.render))

    def on_remove(self, state, hit=False):
        if self.flight is not None and self.flight["state"] is state:
            self.on_motion(state)
            self.flight["ended"] = True
            self.flight["hit"] = bool(hit)
            self.flight["target"] = Vec3(self.sim.enemy.position)

    def before_step(self):
        self.actor_samples = [
            (actor.root, actor.root.getPos(), actor.root.getQuat())
            for actor in (self.sim.player, self.sim.enemy)
        ]

    def _launch_flash(self, position):
        self._particle("glow", position, Vec3(0), 0.18, 4.0, 18.0,
                       (1, 0.7, 0.25, 0.85))
        for _ in range(10):
            self._particle("smoke", position + self._random_direction() * 1.5,
                           self._random_direction() * 4, 1.5, 1.1, 3,
                           (0.84, 0.86, 0.88, 0.45))

    def _random_direction(self):
        z = self.rng.uniform(-1, 1)
        angle = self.rng.uniform(0, math.tau)
        r = math.sqrt(max(0, 1 - z * z))
        return Vec3(r * math.cos(angle), r * math.sin(angle), z)

    def _flight_frame(self, position, dt):
        f = self.flight
        if f["launch_pending"]:
            self._launch_flash(f["previous"])
            f["launch_pending"] = False
        f["age"] += dt
        f["body"].setPos(position)
        # The original AIM-9 is ~11 world metres long. Its origin sits near
        # the centre; the engine plume starts behind it along its actual travel.
        nozzle = position - f["direction"] * 5.3
        pulse = 1 + 0.09 * math.sin(f["age"] * 87)
        for i, flame in enumerate(f["flames"]):
            flame.setPos(nozzle - f["direction"] * i * 0.9)
            flame.setScale((1.35 - i * 0.13) * pulse)
            flame.setColor(1, 0.85 - i * 0.095, 0.36 - i * 0.045,
                           0.95 - i * 0.11)
        f["light"].setPos(nozzle)
        previous = f.get("trail_previous", nozzle)
        travel = nozzle - previous
        distance = travel.length()
        # Space-based emission gives continuous smoke even for 40m physics steps.
        count = min(48, int((distance + f["trail_distance"]) / 1.5))
        for i in range(count):
            sample = previous + travel * ((i + 1) / max(1, count))
            self._particle("smoke", sample, Vec3(0.5, -0.1, 0.9),
                           self.rng.uniform(1.8, 2.5), self.rng.uniform(0.75, 1.2),
                           2.4, (0.88, 0.9, 0.94, 0.58))
        f["trail_distance"] = (distance + f["trail_distance"]) % 1.5
        f["trail_previous"] = Vec3(nozzle)
        self.rendered_flight_frames += 1
        if self.cinematic:
            self._flight_camera(position, f["direction"])

    def _flight_camera(self, position, direction):
        target = self.sim.enemy.root.getPos(self.sim.render)
        centre = position * 0.5 + target * 0.5
        side = direction.cross(Vec3(0, 0, 1))
        if side.lengthSquared() < 0.01:
            side = Vec3(1, 0, 0)
        side.normalize()
        hfov = math.radians(self.sim.camLens.getHfov())
        distance = max(85.0, (target - position).length() * 0.8 /
                       max(0.1, math.tan(hfov * 0.5)))
        camera = centre + side * distance - direction * 26 + Vec3(0, 0, 24)
        self.sim.camera.setPos(camera)
        self.sim.camera.lookAt(centre)

    def _blast(self, position):
        self.impact = Vec3(position)
        self.impact_age = 0.0
        self.blast_count += 1
        self.flash = self._sprite("glow", position, 13, (1, 0.85, 0.55, 1), True)
        self.shockwave = self._sprite("ring", position, 2, (1, 0.8, 0.5, 0.9), True)
        self.blast_light = self._light("impact_light", position, (7, 3, 0.8, 1),
                                      (1, 0.01, 0.002))
        for _ in range(34):
            direction = self._random_direction()
            self._particle("fire", position + direction * self.rng.uniform(0, 4),
                           direction * self.rng.uniform(5, 19),
                           self.rng.uniform(0.65, 1.5), self.rng.uniform(3, 6),
                           self.rng.uniform(5, 11), (1, 0.95, 0.85, 0.95))
        for _ in range(40):
            direction = self._random_direction()
            self._particle("smoke", position + direction * self.rng.uniform(1, 6),
                           direction * self.rng.uniform(3, 10) + Vec3(0, 0, 4),
                           self.rng.uniform(2.5, 4), self.rng.uniform(2, 4),
                           self.rng.uniform(3, 7), (0.20, 0.22, 0.25, 0.8))
        for _ in range(48):
            self._particle("glow", position, self._random_direction() *
                           self.rng.uniform(18, 52), self.rng.uniform(0.6, 1.9),
                           self.rng.uniform(0.18, 0.45), 0.05, (1, 0.5, 0.1, 1))
        for _ in range(18):
            self._particle("debris", position, self._random_direction() *
                           self.rng.uniform(9, 30), self.rng.uniform(2, 3.8),
                           self.rng.uniform(0.3, 0.9), 0, (1, 0.48, 0.12, 1))
        # A visual destruction cue; reset restores the intact aircraft.
        self.sim.enemy.visual_root.hide()
        if self.cinematic:
            self.impact_camera = Vec3(self.sim.camera.getPos(self.sim.render))
            if (self.impact_camera - position).length() < 55:
                self.impact_camera = position + Vec3(65, -65, 28)

    def _clear_flight(self):
        if self.flight is None:
            return
        self.flight["body"].removeNode()
        for node in self.flight["flames"]:
            node.removeNode()
        self._remove_light(self.flight["light"])
        self.flight = None

    def advance(self, dt):
        """Advance only particles, lights and camera, including after terminal."""
        dt = max(0.0, min(0.1, float(dt)))
        kept = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.lifetime:
                p.node.removeNode()
                continue
            progress = p.age / p.lifetime
            if p.kind in ("glow", "debris"):
                p.velocity.z -= 12 * dt
            else:
                p.velocity *= math.exp(-0.6 * dt)
            p.position += p.velocity * dt
            p.node.setPos(p.position)
            p.node.setScale(p.size + p.growth * p.age)
            p.node.setR(p.spin * p.age)
            fade = (1 - progress) ** (1.3 if p.kind == "fire" else 0.8)
            rgb = p.color[:3]
            if p.kind == "fire":
                rgb = (1, max(0.2, 1 - progress * 0.8), max(0.1, 1 - progress))
            elif p.kind == "debris":
                rgb = (max(0.15, 1 - progress * 2), max(0.14, 0.48 - progress), 0.12)
                p.node.setHpr(p.spin * p.age, p.spin * p.age * 0.7, p.spin * p.age * 1.3)
            p.node.setColor(*rgb, p.color[3] * fade)
            kept.append(p)
        self.particles = kept
        if self.impact is None:
            return
        self.impact_age += dt
        age = self.impact_age
        if self.flash is not None:
            self.flash.setScale(13 + age * 60)
            self.flash.setColor(1, 0.83, 0.45, max(0, 1 - age / 0.28))
            if age >= 0.28:
                self.flash.removeNode()
                self.flash = None
        if self.shockwave is not None:
            self.shockwave.setScale(2 + age * 80)
            self.shockwave.setColor(1, 0.8, 0.55, max(0, 0.85 * (1 - age / 0.7)))
            if age >= 0.7:
                self.shockwave.removeNode()
                self.shockwave = None
        if self.blast_light is not None:
            strength = max(0, 1 - age / 1.0)
            self.blast_light.node().setColor((7 * strength, 3 * strength,
                                             0.8 * strength, 1))
            if age >= 1:
                self._remove_light(self.blast_light)
                self.blast_light = None
        if self.cinematic:
            shake = 0.65 * math.exp(-age * 6)
            offset = Vec3(math.sin(age * 73), 0, math.sin(age * 91)) * shake
            self.sim.camera.setPos(self.impact_camera + offset)
            self.sim.camera.lookAt(self.impact + Vec3(0, 0, min(5, age * 2)))

    def _finish_segment(self):
        f = self.flight
        if f is not None and f["ended"]:
            if f["hit"]:
                self._blast(f["target"])
            self._clear_flight()
        elif f is not None:
            f["previous"] = Vec3(f["current"])

    def render_step(self, physics_dt, frame_delay):
        """Render real sample-to-sample motion in slow motion during the shot.

        Only root NodePath transforms are temporarily interpolated. Aircraft
        position/velocity, projectile state, policy actions and dt are untouched.
        """
        f = self.flight
        displayed_duration = max(frame_delay, float(physics_dt) * self.slowmo)
        subdivisions = (max(8, math.ceil(displayed_duration * 60))
                        if f is not None else 1)
        delay = displayed_duration / subdivisions if f is not None else frame_delay
        visual_dt = max(0.0, float(physics_dt)) / subdivisions
        actor_ends = [(node, node.getPos(), node.getQuat(), node.getTransform())
                      for node, _, _ in (self.actor_samples or [])]
        try:
            for sample in range(1, subdivisions + 1):
                alpha = sample / subdivisions
                for start, end in zip(self.actor_samples or [], actor_ends):
                    node, position, quat = start
                    _, end_position, end_quat, _ = end
                    node.setPos(position * (1 - alpha) + end_position * alpha)
                    # Normalized shortest-arc quaternion interpolation is sufficient
                    # for the very short interval between recorded physics samples.
                    if quat.dot(end_quat) < 0:
                        end_quat = -end_quat
                    interpolated = quat * (1 - alpha) + end_quat * alpha
                    interpolated.normalize()
                    node.setQuat(interpolated)
                self.advance(visual_dt)
                if f is not None:
                    position = f["previous"] * (1 - alpha) + f["current"] * alpha
                    self._flight_frame(position, visual_dt)
                self._render_frame()
                if frame_delay > 0:
                    time.sleep(delay)
            self._finish_segment()
        finally:
            for node, _, _, transform in actor_ends:
                node.setTransform(transform)
            self.actor_samples = None

    def _render_frame(self):
        # Pump OS events without taskMgr.step(), which would advance physics twice.
        if self.sim.win is not None and hasattr(self.sim.win, "processEvents"):
            self.sim.win.processEvents()
        self.sim.eventMgr.doEvents()
        self.sim.graphicsEngine.renderFrame()

    def update_realtime(self, dt):
        """Standalone manual JetSim uses the same effects at normal speed."""
        self.advance(dt)
        if self.flight is not None:
            self._flight_frame(self.flight["current"], dt)
        self._finish_segment()

    def hold(self, seconds, frame_delay):
        # Always allow a confirmed hit's four-second blast to finish.
        remaining = max(0, self.BLAST_DURATION - self.impact_age) if self.impact is not None else 0
        deadline = time.monotonic() + max(0, seconds, remaining)
        previous = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            self.advance(now - previous)
            previous = now
            self._render_frame()
            time.sleep(frame_delay if frame_delay > 0 else 1 / 60)

    def reset(self):
        self._clear_flight()
        for p in self.particles:
            p.node.removeNode()
        self.particles.clear()
        for node in (self.flash, self.shockwave):
            if node is not None:
                node.removeNode()
        self._remove_light(self.blast_light)
        self.blast_light = self.flash = self.shockwave = None
        self.impact = None
        self.impact_age = 0.0
        self.actor_samples = None
        self.sim.enemy.visual_root.show()

    def destroy(self):
        self.reset()
        self.root.removeNode()
