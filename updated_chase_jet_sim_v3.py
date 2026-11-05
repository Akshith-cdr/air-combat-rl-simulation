from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Vec3, AmbientLight, DirectionalLight, Filename,
    NodePath, Material, Quat, TextNode, TransparencyAttrib,
    GeomVertexFormat, GeomVertexData, Geom, GeomTriangles,
    GeomVertexWriter, GeomNode
)
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import math
import random
import sys
from pathlib import Path
from projectile_motion import (
    linear_position, sphere_crossings, swept_contact_fraction, validate_dt,
    GuidanceConfig, GuidedMissile, TargetTrack, advance_guided_salvo,
)
from radar import RadarConfig, RadarContact, PandaRadar, project_contacts
from weapon_config import launch_distance_allowed

# ============================================================
# Portable asset loading
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

def asset_path(name: str) -> str:
    return str(BASE_DIR / name)

# ============================================================
# World / gameplay settings
# ============================================================
NUM_CLOUDS = 250
CLOUD_AREA_RADIUS = 20000
CLOUD_MIN_ALTITUDE = 50
CLOUD_MAX_ALTITUDE = 3500
CLOUD_MIN_SIZE = 15
CLOUD_MAX_SIZE = 200

# Player aircraft
PLAYER_MODEL_PATH = "ba_f-15ex_eagle_ii.glb"
PLAYER_VISUAL_FIX_HPR = Vec3(0, 90, 180)
PLAYER_SCALE = 110
PLAYER_START_POS = Vec3(0, 0, 1000)
PLAYER_START_HPR = Vec3(0, 0, 0)

# Enemy aircraft
ENEMY_MODEL_PATH = "enemy.glb"
ENEMY_VISUAL_FIX_HPR = Vec3(90, 90, 0)
ENEMY_SCALE = 0.1
ENEMY_START_POS = Vec3(0, 3000, 1200)
ENEMY_START_HPR = Vec3(180, 0, 0)

# Missile
MISSILE_MODEL_PATH = "aim-9_missile.glb"
MISSILE_RELATIVE_SCALE = 0.05
MISSILE_MOUNT_POS = Vec3(0.0, 0.0, 0.09)
MISSILE_MOUNT_HPR = Vec3(0.0, 0.0, 0.0)
MISSILE_LAUNCH_SPEED = 35.0
MISSILE_MAX_LIFETIME = 20.0
MISSILE_MAX_RANGE = 25000.0
# Collision capture radius, applied continuously over each linear frame segment.
MISSILE_HIT_RADIUS = 50.0


def apply_missile_appearance(node):
    """Project's existing gray material; retain the original embedded textures.

    The previous setTexture(stage, None) attempt raises TypeError in Panda3D,
    so textures historically remained active. No new paint or tint is added.
    """
    material = Material()
    material.setDiffuse((0.62, 0.62, 0.62, 1.0))
    material.setAmbient((0.62, 0.62, 0.62, 1.0))
    material.setSpecular((0.10, 0.10, 0.10, 1.0))
    material.setShininess(6.0)
    node.setMaterial(material, 1)
    node.setColorScale(1, 1, 1, 1)
    node.setTwoSided(True)

# Physics
GRAVITY = 0.0
AIR_DENSITY_SEA_LEVEL = 1.225
DRAG_COEFFICIENT = 0.00
LIFT_COEFFICIENT = 10.2
WING_AREA = 50.0
MASS = 1800.0
MAX_SPEED = 800.0
MAX_G_FORCE = 9.0
MAX_ROLL = 90.0  # Change this value to control the aircraft's maximum bank angle

# Control sensitivity
PITCH_SENSITIVITY = 40.5
ROLL_SENSITIVITY = 51.0
YAW_SENSITIVITY = 40.3

# Lower response = less twitchy
CONTROL_RESPONSE = 8.0

# Turn coordination
MIN_ROLL_FOR_TURN = 15.0
MAX_TURN_RATE = 285.0
TURN_COORDINATION_FACTOR = 0.35

# Speed control
SPEED_CHANGE_RATE = 100.0
SPEED_RESPONSE = 1.6
TURN_DRAG_RATE = 10.0
DIR_BLEND_RATE = 2.5

# Boundary around enemy (moving spherical chase boundary)
CHASE_BOUNDARY_RADIUS = 10000.0

# Enemy AI. The patrol envelope is intentionally three-dimensional so the
# pursuer must solve both horizontal and vertical interception geometry.
ENEMY_PATROL_RADIUS = 1500.0
ENEMY_PATROL_VERTICAL_RADIUS = 900.0
ENEMY_PATROL_MIN_ALTITUDE = 350.0
ENEMY_PATROL_MAX_ALTITUDE = 3200.0
ENEMY_MIN_SPEED = 140.0
ENEMY_MAX_SPEED = 240.0
ENEMY_WAYPOINT_REACH_DIST = 120.0
ENEMY_AI_UPDATE_TIME_MIN = 2.0
ENEMY_AI_UPDATE_TIME_MAX = 5.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def vec_length_sq(v):
    return v.x * v.x + v.y * v.y + v.z * v.z


def wrap_angle_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def shortest_angle_delta(target, current):
    return wrap_angle_deg(target - current)


def random_point_in_sphere(radius, rng):
    while True:
        p = Vec3(
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
        )
        if vec_length_sq(p) <= 1.0:
            if vec_length_sq(p) > 1e-8:
                p.normalize()
            p *= radius * (rng.random() ** (1.0 / 3.0))
            return p


class Cloud:
    def __init__(self, showbase, position, size, rng):
        self.showbase = showbase
        self.position = position
        self.size = size
        self.rng = rng
        self.node = None
        self.create_cloud()

    def create_cloud(self):
        cloud_node = NodePath("cloud")
        num_puffs = self.rng.randint(3, 5)
        for _ in range(num_puffs):
            puff = self.create_cloud_puff()
            puff.setPos(
                self.rng.uniform(-self.size * 0.3, self.size * 0.3),
                self.rng.uniform(-self.size * 0.3, self.size * 0.3),
                self.rng.uniform(-self.size * 0.1, self.size * 0.2),
            )
            puff.reparentTo(cloud_node)

        cloud_node.setPos(self.position)
        cloud_node.setScale(self.size)
        cloud_node.setTransparency(TransparencyAttrib.MAlpha)
        cloud_node.setAlphaScale(0.7)

        self.node = cloud_node
        cloud_node.reparentTo(self.showbase.render)

    def create_cloud_puff(self):
        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('cloud_puff', fmt, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')

        r = self.rng.uniform(0.85, 0.95)
        g = self.rng.uniform(0.85, 0.95)
        b = self.rng.uniform(0.85, 0.95)

        segments = 12
        rings = 8

        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)

                x = cos_theta * sin_phi
                y = sin_theta * sin_phi
                z = cos_phi

                vertex.addData3f(x, y, z)
                normal.addData3f(x, y, z)
                color.addData4f(r, g, b, 0.7)

        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHStatic)

        for ring in range(rings):
            for seg in range(segments):
                a = ring * (segments + 1) + seg
                b = a + 1
                c = a + (segments + 1)
                d = c + 1
                tris.addVertices(a, b, c)
                tris.addVertices(b, d, c)

        geom.addPrimitive(tris)
        node = GeomNode('cloud_puff')
        node.addGeom(geom)
        return self.showbase.render.attachNewNode(node)


class AircraftActor:
    def __init__(
        self,
        sim,
        name,
        model_path,
        initial_pos,
        initial_hpr,
        scale,
        visual_fix_hpr,
        use_missile_mount=False,
    ):
        self.sim = sim
        self.name = name

        # Physics root: all game movement goes here.
        self.root = sim.render.attachNewNode(f"{name}_root")

        # Visual root: model-only fixups happen here.
        self.visual_root = self.root.attachNewNode(f"{name}_visual_root")

        if getattr(sim, "headless", False):
            self.model = NodePath(name + "_headless")
        else:
            try:
                self.model = sim.loader.loadModel(Filename.fromOsSpecific(asset_path(model_path)))
            except Exception as e:
                raise RuntimeError(f"Error loading {name} model '{model_path}': {e}")

        self.model.reparentTo(self.visual_root)

        # Normalize the visual model once so exported model quirks do not affect physics.
        self.visual_root.setPos(0, 0, 0)
        self.visual_root.setHpr(visual_fix_hpr)
        self.visual_root.setScale(scale)

        self.root.setPos(initial_pos)
        self.root.setHpr(initial_hpr)

        try:
            ts_list = self.model.findAllTextureStages()
            for ts in ts_list:
                self.model.setTexture(ts, None)
        except Exception:
            pass

        try:
            mat = Material()
            mat.setDiffuse((0.5, 0.5, 0.5, 1.0))
            mat.setSpecular((0.08, 0.08, 0.08, 1.0))
            mat.setShininess(8.0)
            self.model.setMaterial(mat, 1)
            self.model.setColorScale(1, 1, 1, 1)
            self.model.setTwoSided(True)
        except Exception:
            pass

        self.position = Vec3(initial_pos)
        self.velocity = Vec3(0, 150, 0)
        self.roll = initial_hpr.x
        self.pitch = initial_hpr.y
        self.yaw = initial_hpr.z

        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        self.pitch_input = 0.0
        self.roll_input = 0.0
        self.yaw_input = 0.0
        self.speed_input = 0.0

        self.target_speed = 150.0
        self.current_speed = 150.0
        self.engine_on = True

        self.airspeed = 150.0
        self.altitude = max(0.0, initial_pos.z)
        self.g_force = 1.0
        self.turn_rate = 0.0
        self.target_turn_rate = 0.0

        self.crashed = False
        self.crash_message = ""

        self.autopilot = False
        self.ai_waypoint = Vec3(initial_pos)
        self.ai_waypoint_timer = 0.0
        self.ai_patrol_center = Vec3(initial_pos)
        self.ai_update_timer = 0.0

        self.use_missile_mount = use_missile_mount
        self.missile_mount_node = None
        if use_missile_mount:
            self.missile_mount_node = self.visual_root.attachNewNode(f"{name}_missile_mount")
            self.missile_mount_node.setPos(MISSILE_MOUNT_POS)
            self.missile_mount_node.setHpr(MISSILE_MOUNT_HPR)

        self.boundary_sphere = None

    def sync_graphics(self):
        safe_pos = Vec3(
            0.0 if math.isnan(self.position.x) or math.isinf(self.position.x) else self.position.x,
            0.0 if math.isnan(self.position.y) or math.isinf(self.position.y) else self.position.y,
            1000.0 if math.isnan(self.position.z) or math.isinf(self.position.z) else max(0.0, self.position.z),
        )
        self.root.setPos(safe_pos)

        safe_h = 0.0 if math.isnan(self.yaw) or math.isinf(self.yaw) else self.yaw
        safe_p = 0.0 if math.isnan(self.pitch) or math.isinf(self.pitch) else self.pitch
        safe_r = 0.0 if math.isnan(self.roll) or math.isinf(self.roll) else self.roll
        self.root.setHpr(safe_h, safe_p, safe_r)

    def get_forward_vector(self):
        quat = Quat()
        quat.setHpr(Vec3(self.yaw, self.pitch, self.roll))
        forward = quat.xform(Vec3(0, 1, 0))
        if vec_length_sq(forward) > 0:
            forward.normalize()
        return forward

    def get_up_vector(self):
        quat = Quat()
        quat.setHpr(Vec3(self.yaw, self.pitch, self.roll))
        up = quat.xform(Vec3(0, 0, 1))
        if vec_length_sq(up) > 0:
            up.normalize()
        return up


class JetSim(ShowBase):
    def __init__(self, enable_missile_vfx=False, guidance_config=None, headless=False,
                 radar_config=None):
        self.headless = bool(headless)
        self.missile_presentation = True
        self.player_camera_locked = False
        self.camera_target = "player"
        self.guidance_config = guidance_config
        self.radar_config = radar_config or RadarConfig()
        self.radar = None
        self.radar_contacts = {}
        self.missile_events = []
        self._missile_sequence = 0
        self._physics_accumulator = 0.0
        self.rng = random.Random()
        if self.headless:
            # Scene graph transforms only: no ShowBase, GPU, window, tasks or assets.
            self.render = NodePath("headless_simulation")
            self.player = AircraftActor(self, "player", PLAYER_MODEL_PATH, PLAYER_START_POS,
                                        PLAYER_START_HPR, PLAYER_SCALE, PLAYER_VISUAL_FIX_HPR, True)
            self.enemy = AircraftActor(self, "enemy", ENEMY_MODEL_PATH, ENEMY_START_POS,
                                       ENEMY_START_HPR, ENEMY_SCALE, ENEMY_VISUAL_FIX_HPR)
            self.enemy.autopilot = True
            self.missile = self.player.missile_mount_node.attachNewNode("mounted_missile")
            self.missile_ready = True
            self.active_missiles = []
            self.missile_hit = False
            self.missile_hit_count = 0
            self.target_destroyed = False
            self.missile_vfx = None
            self.game_over = False
            self.game_over_message = ""
            return
        super().__init__()

        self.render.setShaderAuto()
        self.disableMouse()
        self.setBackgroundColor(0.53, 0.80, 0.92, 1)

        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.45, 0.45, 1))
        alnp = self.render.attachNewNode(ambient)
        self.render.setLight(alnp)

        dlight = DirectionalLight("dlight")
        dlight.setColor((0.9, 0.9, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-30, -60, 0)
        self.render.setLight(dlnp)

        self.clouds = []
        self.create_clouds()

        self.player = AircraftActor(
            sim=self,
            name="player",
            model_path=PLAYER_MODEL_PATH,
            initial_pos=PLAYER_START_POS,
            initial_hpr=PLAYER_START_HPR,
            scale=PLAYER_SCALE,
            visual_fix_hpr=PLAYER_VISUAL_FIX_HPR,
            use_missile_mount=True,
        )

        self.enemy = AircraftActor(
            sim=self,
            name="enemy",
            model_path=ENEMY_MODEL_PATH,
            initial_pos=ENEMY_START_POS,
            initial_hpr=ENEMY_START_HPR,
            scale=ENEMY_SCALE,
            visual_fix_hpr=ENEMY_VISUAL_FIX_HPR,
            use_missile_mount=False,
        )
        self.enemy.autopilot = True
        self.enemy.ai_patrol_center = Vec3(ENEMY_START_POS)
        self.enemy.ai_waypoint = Vec3(ENEMY_START_POS)

        self.load_boundary_sphere()

        self.missile = None
        self.attach_missile()
        self.missile_ready = self.missile is not None
        self.active_missiles = []
        self.missile_hit = False
        self.missile_hit_count = 0
        self.target_destroyed = False
        self.missile_vfx = None
        if enable_missile_vfx:
            self.enable_missile_effects(cinematic=False)

        self.camera_offset = Vec3(0, -30, 10)
        self.camera_look_offset = Vec3(0, 50, 0)
        self.camera_mode = 0

        self.game_over = False
        self.game_over_message = ""

        self.hud_text = None
        self.create_hud()
        self.radar = PandaRadar(self.a2dBottomRight)

        self.setup_controls()
        self.taskMgr.add(self.update_task, "update_task")

    # ========================================================
    # Scene setup
    # ========================================================
    def set_seed(self, seed):
        self.rng.seed(seed)

    def create_clouds(self):
        print(f"Creating {NUM_CLOUDS} clouds...")
        for _ in range(NUM_CLOUDS):
            angle = self.rng.uniform(0, 2 * math.pi)
            distance = self.rng.uniform(0, CLOUD_AREA_RADIUS)
            x = math.cos(angle) * distance
            y = math.sin(angle) * distance
            z = self.rng.uniform(CLOUD_MIN_ALTITUDE, CLOUD_MAX_ALTITUDE)
            size = self.rng.uniform(CLOUD_MIN_SIZE, CLOUD_MAX_SIZE)
            cloud = Cloud(self, Vec3(x, y, z), size, self.rng)
            self.clouds.append(cloud)
        print("Clouds created successfully!")

    def load_boundary_sphere(self):
        try:
            sphere = self.loader.loadModel("models/misc/sphere")
            sphere.reparentTo(self.enemy.root)
            sphere.setScale(CHASE_BOUNDARY_RADIUS)
            sphere.setPos(0, 0, 0)
            sphere.setTransparency(TransparencyAttrib.MAlpha)
            sphere.setColor(1.0, 0.4, 0.1, 0.08)
            sphere.setTwoSided(True)
            self.enemy.boundary_sphere = sphere
        except Exception as e:
            print("Boundary sphere could not be loaded:", e)
            self.enemy.boundary_sphere = None

    def create_hud(self):
        self.hud_text = OnscreenText(
            text="",
            pos=(-1.8, 0.9),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

    # ========================================================
    # Controls
    # ========================================================
    def setup_controls(self):
        self.accept("escape", sys.exit)

        self.accept("arrow_up", self.set_player_pitch_input, [1.0])
        self.accept("arrow_up-up", self.set_player_pitch_input, [0.0])
        self.accept("arrow_down", self.set_player_pitch_input, [-1.0])
        self.accept("arrow_down-up", self.set_player_pitch_input, [0.0])

        self.accept("arrow_left", self.set_player_roll_input, [1.0])
        self.accept("arrow_left-up", self.set_player_roll_input, [0.0])
        self.accept("arrow_right", self.set_player_roll_input, [-1.0])
        self.accept("arrow_right-up", self.set_player_roll_input, [0.0])

        self.accept("q", self.set_player_yaw_input, [1.0])
        self.accept("q-up", self.set_player_yaw_input, [0.0])
        self.accept("e", self.set_player_yaw_input, [-1.0])
        self.accept("e-up", self.set_player_yaw_input, [0.0])

        self.accept("w", self.set_player_speed_input, [1.0])
        self.accept("w-up", self.set_player_speed_input, [0.0])
        self.accept("s", self.set_player_speed_input, [-1.0])
        self.accept("s-up", self.set_player_speed_input, [0.0])

        self.accept("t", self.toggle_player_engine)
        self.accept("c", self.cycle_camera_mode)
        self.accept("m", self.fire_missile)

    def set_player_pitch_input(self, value):
        if not self.game_over:
            self.player.pitch_input = value

    def set_player_roll_input(self, value):
        if not self.game_over:
            self.player.roll_input = value

    def set_player_yaw_input(self, value):
        if not self.game_over:
            self.player.yaw_input = value

    def set_player_speed_input(self, value):
        if not self.game_over:
            self.player.speed_input = value

    def toggle_player_engine(self):
        if not self.game_over:
            self.player.engine_on = not self.player.engine_on
            if not self.player.engine_on:
                self.player.target_speed = 0.0

    def cycle_camera_mode(self):
        if not self.game_over and not self.player_camera_locked:
            self.camera_mode = (self.camera_mode + 1) % 2

    def configure_presentation(self, *, missiles=True, lock_player_camera=False):
        """Set rendering only; inventory and numerical projectile states are untouched.

        Radar-only mode removes missile geometry, rather than merely hiding it.
        Empty transform nodes retain compatibility with existing event interfaces.
        """
        self.missile_presentation = bool(missiles)
        self.player_camera_locked = bool(lock_player_camera)
        if not self.missile_presentation:
            if self.missile_vfx is not None:
                self.missile_vfx.destroy()
                self.missile_vfx = None
            nodes = [(None, self.missile)] + [(state, state["node"]) for state in self.active_missiles]
            for state, node in nodes:
                if node is None or node.isEmpty():
                    continue
                replacement = node.getParent().attachNewNode("numerical_missile")
                replacement.setTransform(node.getTransform())
                node.removeNode()
                if state is None:
                    self.missile = replacement
                else:
                    state["node"] = replacement
        if self.player_camera_locked:
            self.camera_mode = 0
            self.camera_target = "player"
        self.update_camera()

    # ========================================================
    # Missile
    # ========================================================
    def enable_missile_effects(self, cinematic=True, slowmo=8.0):
        if not self.missile_presentation:
            return None
        from missile_vfx import MissileVFX

        if self.headless:
            raise ValueError("VFX requires a rendered simulator")
        if self.missile_vfx is not None:
            self.missile_vfx.destroy()
        self.missile_vfx = MissileVFX(self, cinematic=cinematic and not self.player_camera_locked,
                                      slowmo=slowmo)
        return self.missile_vfx

    def destroy(self):
        if getattr(self, "_stage1_destroyed", False):
            return
        self._stage1_destroyed = True
        simple = getattr(self, "simple_missile_visuals", None)
        if simple is not None:
            simple.destroy()
            self.simple_missile_visuals = None
        effects = getattr(self, "missile_vfx", None)
        if effects is not None:
            effects.destroy()
            self.missile_vfx = None
        if getattr(self, "radar", None) is not None:
            self.radar.destroy()
            self.radar = None
        if getattr(self, "headless", False):
            self.render.removeNode()
        else:
            super().destroy()

    def attach_missile(self):
        self.missile = self._create_missile_node()

        self.missile.reparentTo(self.player.missile_mount_node)

        apply_missile_appearance(self.missile)

        self.missile.setScale(MISSILE_RELATIVE_SCALE)
        self.missile.setPos(0, 0, 0)
        self.missile.setHpr(0, 0, 0)

    def _create_missile_node(self):
        """A missing presentation asset must not disable a numerical weapon."""
        if not getattr(self, "headless", False) and getattr(self, "missile_presentation", True):
            try:
                node = self.loader.loadModel(Filename.fromOsSpecific(asset_path(MISSILE_MODEL_PATH)))
                apply_missile_appearance(node)
                return node
            except (OSError, IOError) as exc:
                print("Missile presentation asset unavailable:", exc)
        return NodePath("missile")

    def load_missile_presentation_model(self):
        """Cache the original asset before flight, at its original world scale.

        The asset nose is local +Z. A -90-degree pitch maps it into the +Y
        tracking container, whose heading follows numerical missile velocity.
        Missing assets are reported, never silently replaced with a new shape.
        """
        if self.headless:
            return None
        path = Path(asset_path(MISSILE_MODEL_PATH))
        if not path.is_file():
            raise FileNotFoundError(f"Original missile asset unavailable: {path}")
        node = self.loader.loadModel(Filename.fromOsSpecific(str(path)))
        node.setName("original_missile_model")
        apply_missile_appearance(node)
        node.setScale(PLAYER_SCALE * MISSILE_RELATIVE_SCALE)
        node.setHpr(0, -90, 0)
        return node

    def fire_missile(self):
        if self.game_over:
            return
        if (self.target_destroyed or self.enemy.crashed
                or not launch_distance_allowed(
                    (self.enemy.position - self.player.position).length())):
            return
        if not getattr(self, "missile_ready", True):
            return
        if self.missile is None:
            print("Missile model is not available.")
            return
        if not hasattr(self, "missile_ready") and self.missile.isHidden():
            return

        if getattr(self, "guidance_config", None) is not None:
            # Authorization/inventory remains here; guidance receives numerical state.
            direction = self.player.get_forward_vector()
            launch_velocity = tuple(self.player.velocity + direction * MISSILE_LAUNCH_SPEED)
            # Aircraft can reach 800 m/s; GuidedMissile requires <=650 m/s by
            # default. Respect its existing cap instead of crashing on launch.
            launch_speed = math.hypot(*launch_velocity)
            if launch_speed > self.guidance_config.max_speed:
                # Scale in double precision: Panda's float32 Vec3 can round
                # a normalized diagonal velocity back above the strict cap.
                scale = math.nextafter(self.guidance_config.max_speed, 0.0) / launch_speed
                launch_velocity = tuple(component * scale for component in launch_velocity)
            state = self.launch_guided_missile(
                "friendly", "enemy", tuple(self.player.position),
                launch_velocity)
            if state is not None:
                self.missile.hide()
                self.missile_ready = False
            return

        # Preserve the historical mount offset and launch axis numerically.
        # The old mesh's local +Z launch direction is intentionally legacy-only.
        if hasattr(self.player, "position"):
            aircraft_q, visual_q, mount_q = Quat(), Quat(), Quat()
            aircraft_q.setHpr(Vec3(self.player.yaw, self.player.pitch, self.player.roll))
            visual_q.setHpr(PLAYER_VISUAL_FIX_HPR)
            mount_q.setHpr(MISSILE_MOUNT_HPR)
            launch_pos = self.player.position + aircraft_q.xform(
                visual_q.xform(MISSILE_MOUNT_POS * PLAYER_SCALE))
            launch_direction = aircraft_q.xform(visual_q.xform(mount_q.xform(Vec3(0, 0, 1))))
            launch_hpr = (aircraft_q * visual_q * mount_q).getHpr()
        else:
            # Compatibility for old scene-only callers; new launches use actors.
            launch_pos = self.missile.getPos(self.render)
            launch_hpr = self.missile.getHpr(self.render)
            q = Quat()
            q.setHpr(launch_hpr)
            launch_direction = q.xform(Vec3(0, 0, 1))

        projectile = JetSim._create_missile_node(self)
        projectile.reparentTo(self.render)
        projectile.setPos(launch_pos)
        projectile.setHpr(launch_hpr)

        world_scale = self.player.visual_root.getScale(self.render).x * MISSILE_RELATIVE_SCALE
        projectile.setScale(world_scale)

        if vec_length_sq(launch_direction) < 1e-8:
            launch_direction = Vec3(0, 0, 1)
        else:
            launch_direction.normalize()

        projectile_state = {
            "node": projectile,
            "position": Vec3(launch_pos),
            "missile_id": f"missile-{getattr(self, '_missile_sequence', 0)}",
            "team": "friendly",
            "target_id": "enemy",
            "velocity": launch_direction * MISSILE_LAUNCH_SPEED + self.player.velocity,
            "age": 0.0,
            "spawn_pos": Vec3(launch_pos),
            "hit_registered": False,
        }
        self._missile_sequence = getattr(self, "_missile_sequence", 0) + 1
        self.active_missiles.append(projectile_state)
        if self.missile_vfx is not None:
            self.missile_vfx.on_launch(projectile_state)
        self.missile.hide()
        self.missile_ready = False
        print("Missile launched.")

    def launch_guided_missile(self, team, target_id, position, velocity, config=None):
        """Numerical launch interface for authorized callers; no AI or ammo policy.

        Both teams and any number of projectiles are supported. The existing
        player fire interface still consumes exactly one mounted round.
        """
        if team not in ("friendly", "enemy") or target_id not in ("player", "enemy"):
            raise ValueError("unknown team or target")
        if (team == "friendly") != (target_id == "enemy"):
            raise ValueError("friendly fire is not supported")
        target = self.enemy if target_id == "enemy" else self.player
        if self.game_over or target.crashed or (target_id == "enemy" and self.target_destroyed):
            return None
        launcher = self.player if team == "friendly" else self.enemy
        # Use actor world positions, never the supplied projectile spawn/mount.
        if not launch_distance_allowed((target.position - launcher.position).length()):
            return None
        identity = f"missile-{self._missile_sequence}"
        guided = GuidedMissile(identity, team, target_id, position, velocity,
                               config or self.guidance_config or GuidanceConfig())
        self._missile_sequence += 1
        node = self._create_missile_node()
        node.setName(identity)
        node.reparentTo(self.render)
        node.setScale(PLAYER_SCALE * MISSILE_RELATIVE_SCALE)
        node.setPos(*guided.position)
        state = {"node": node, "guided": guided, "position": Vec3(*guided.position),
                 "velocity": Vec3(*guided.velocity), "age": 0.0,
                 "spawn_pos": Vec3(*guided.position), "hit_registered": False,
                 "missile_id": identity, "team": team, "target_id": target_id}
        self.active_missiles.append(state)
        if self.missile_vfx is not None:
            self.missile_vfx.on_launch(state)
        return state

    def register_missile_hit(self, missile_state):
        """Register one projectile hit and expose the event for future consumers."""
        if missile_state.get("hit_registered", False):
            return

        target_id = missile_state.get("target_id", "enemy")
        target = self.enemy if target_id == "enemy" else self.player
        if target_id == "player":
            if target.crashed:
                return
            missile_state["hit_registered"] = True
            target.crashed = True
            target.crash_message = "MISSILE HIT PLAYER"
            target.velocity = Vec3(0)
            target.roll_rate = target.pitch_rate = target.yaw_rate = target.turn_rate = 0.0
            self.game_over = True
            self.game_over_message = "MISSILE HIT PLAYER"
            return
        if self.target_destroyed:
            return

        missile_state["hit_registered"] = True
        self.missile_hit = True
        self.missile_hit_count += 1
        self.target_destroyed = True
        self.enemy.crashed = True
        self.enemy.crash_message = "MISSILE HIT TARGET"
        self.enemy.velocity = Vec3(0, 0, 0)
        self.enemy.roll_rate = self.enemy.pitch_rate = self.enemy.yaw_rate = 0.0
        self.enemy.turn_rate = 0.0
        print("MISSILE HIT TARGET")

    def update_missiles(self, dt, target_previous_position=None, player_previous_position=None):
        """Advance legacy or guided numerical states; apply damage after contact.

        The optional position is the target's actual start of this physics
        frame, not a prediction. Pass both starts when either team can fire.
        Callers omitting a start get a stationary-target sweep for that aircraft.
        This consumes the caller's dt without reading the rendering clock.
        """
        dt = validate_dt(dt)
        if not self.active_missiles:
            return

        target_end = Vec3(self.enemy.position)
        target_start = (Vec3(target_previous_position)
                        if target_previous_position is not None else target_end)
        guided_states = [m["guided"] for m in self.active_missiles if "guided" in m]
        guided_events = {}
        if guided_states:
            tracks = {
                "enemy": TargetTrack("enemy", "enemy", tuple(self.enemy.position),
                                     tuple(self.enemy.velocity),
                                     not self.enemy.crashed and not self.target_destroyed),
                "player": TargetTrack("player", "friendly", tuple(self.player.position),
                                      tuple(self.player.velocity), not self.player.crashed),
            }
            previous = {"enemy": tuple(target_start),
                        "player": tuple(player_previous_position) if player_previous_position is not None
                        else tuple(self.player.position)}
            events = advance_guided_salvo(guided_states, tracks, previous, dt)
            self.missile_events.extend(events)
            guided_events = {event.missile_id: event for event in events}
        kept = []
        for m in self.active_missiles:
            if "guided" in m:
                guided = m["guided"]
                event = guided_events.get(guided.missile_id)
                if event is None or event.reason != "invalid_state":
                    m.update(position=Vec3(*guided.position), velocity=Vec3(*guided.velocity), age=guided.age)
                    m["node"].setPos(m["position"])
                    if m["velocity"].lengthSquared() > 1e-12:
                        m["node"].lookAt(m["position"] + m["velocity"])
                        m["node"].setP(m["node"].getP() - 90)
                    if self.missile_vfx is not None:
                        self.missile_vfx.on_motion(m)
                if event is not None:
                    if event.reason == "hit":
                        self.register_missile_hit(m)
                    self._remove_projectile(m, hit=event.reason == "hit")
                else:
                    kept.append(m)
                continue
            node = m["node"]
            start = Vec3(m["position"] if "position" in m else node.getPos())
            # Already-expired projectiles cannot cause a new contact event.
            if (m["age"] >= MISSILE_MAX_LIFETIME
                    or (start - m["spawn_pos"]).length() >= MISSILE_MAX_RANGE
                    or start.z < -1000):
                self._remove_projectile(m)
                continue

            duration = min(dt, MISSILE_MAX_LIFETIME - m["age"])
            end = Vec3(*linear_position(start, m["velocity"], duration))
            limit = 1.0
            boundary_reached = False
            exits = sphere_crossings(start, end, m["spawn_pos"], MISSILE_MAX_RANGE)
            if exits is not None and 0.0 <= exits[1] <= limit:
                limit = exits[1]
                boundary_reached = True
            if end.z <= -1000 and end.z != start.z:
                limit = min(limit, (-1000 - start.z) / (end.z - start.z))
                boundary_reached = True
            elapsed = duration * limit
            end = Vec3(*linear_position(start, m["velocity"], elapsed))
            target_segment_end = target_start + (target_end - target_start) * (
                elapsed / dt if dt > 0 else 0.0
            )
            contact = None if self.target_destroyed else swept_contact_fraction(
                start, end, target_start, target_segment_end, MISSILE_HIT_RADIUS
            )
            if contact is not None:
                elapsed *= contact
                end = Vec3(*linear_position(start, m["velocity"], elapsed))
            m["age"] += elapsed
            if "position" in m:
                m["position"] = Vec3(end)
            node.setPos(end)
            if self.missile_vfx is not None:
                self.missile_vfx.on_motion(m)

            if contact is not None:
                self.register_missile_hit(m)
                self._remove_projectile(m, hit=True)
                continue

            if elapsed < dt or m["age"] >= MISSILE_MAX_LIFETIME or boundary_reached:
                self._remove_projectile(m)
                continue

            kept.append(m)

        self.active_missiles = kept

    def _remove_projectile(self, state, hit=False):
        if self.missile_vfx is not None:
            self.missile_vfx.on_remove(state, hit=hit)
        state["node"].removeNode()

    # ========================================================
    # Physics helpers
    # ========================================================
    def calculate_lift(self, airspeed, angle_of_attack):
        q = 0.5 * AIR_DENSITY_SEA_LEVEL * airspeed ** 2
        alpha_rad = math.radians(angle_of_attack)
        cl = LIFT_COEFFICIENT * math.sin(2 * alpha_rad)
        lift = q * WING_AREA * cl
        if abs(angle_of_attack) > 15:
            lift *= 0.3
        return lift

    def calculate_drag(self, airspeed, angle_of_attack):
        q = 0.5 * AIR_DENSITY_SEA_LEVEL * airspeed ** 2
        alpha_rad = math.radians(angle_of_attack)
        cd = DRAG_COEFFICIENT + 0.1 * abs(alpha_rad)
        return q * WING_AREA * cd

    def is_nan_vector(self, vec):
        return (
            math.isnan(vec.x) or math.isnan(vec.y) or math.isnan(vec.z) or
            math.isinf(vec.x) or math.isinf(vec.y) or math.isinf(vec.z)
        )

    def is_nan_value(self, value):
        return math.isnan(value) or math.isinf(value)

    def sanitize_actor(self, actor: AircraftActor):
        if self.is_nan_vector(actor.position):
            actor.position = Vec3(0, 0, 1000)

        if self.is_nan_vector(actor.velocity):
            actor.velocity = Vec3(0, 150, 0)

        if self.is_nan_value(actor.roll):
            actor.roll = 0.0
        if self.is_nan_value(actor.pitch):
            actor.pitch = 0.0
        if self.is_nan_value(actor.yaw):
            actor.yaw = 0.0

        if self.is_nan_value(actor.roll_rate):
            actor.roll_rate = 0.0
        if self.is_nan_value(actor.pitch_rate):
            actor.pitch_rate = 0.0
        if self.is_nan_value(actor.yaw_rate):
            actor.yaw_rate = 0.0

        if self.is_nan_value(actor.current_speed):
            actor.current_speed = 150.0
        if self.is_nan_value(actor.target_speed):
            actor.target_speed = 150.0
        if self.is_nan_value(actor.airspeed):
            actor.airspeed = 150.0

        actor.pitch = clamp(actor.pitch, -90, 90)
        actor.roll = wrap_angle_deg(actor.roll)
        actor.yaw = actor.yaw % 360
        actor.current_speed = clamp(actor.current_speed, 0.0, MAX_SPEED)
        actor.target_speed = clamp(actor.target_speed, 0.0, MAX_SPEED)

    def update_speed_control(self, actor: AircraftActor, dt):
        if not actor.engine_on:
            actor.target_speed = max(0.0, actor.target_speed - SPEED_CHANGE_RATE * dt)
        else:
            if actor.speed_input > 0:
                actor.target_speed += SPEED_CHANGE_RATE * dt
            elif actor.speed_input < 0:
                actor.target_speed -= SPEED_CHANGE_RATE * dt

        actor.target_speed = clamp(actor.target_speed, 0.0, MAX_SPEED)

        # Smooth speed tracking. This uses actual speed only, not forward-projection,
        # so turning does not cause a speed surge.
        actor.current_speed += (actor.target_speed - actor.current_speed) * SPEED_RESPONSE * dt

        # Very small turn drag to avoid "power-on" acceleration while banking.
        actor.current_speed -= abs(actor.roll) / max(1.0, MAX_TURN_RATE) * TURN_DRAG_RATE * dt
        actor.current_speed = clamp(actor.current_speed, 0.0, MAX_SPEED)

    def calculate_turn_coordination(self, actor: AircraftActor, dt):
        roll_abs = abs(actor.roll)
        if roll_abs < MIN_ROLL_FOR_TURN:
            actor.target_turn_rate = 0.0
            actor.turn_rate = 0.0
            return

        roll_excess = roll_abs - MIN_ROLL_FOR_TURN
        roll_normalized = min(1.0, roll_excess / max(1.0, (MAX_ROLL - MIN_ROLL_FOR_TURN)))
        actor.target_turn_rate = roll_normalized * MAX_TURN_RATE
        if actor.roll > 0:
            actor.target_turn_rate = -actor.target_turn_rate

        actor.turn_rate += (actor.target_turn_rate - actor.turn_rate) * TURN_COORDINATION_FACTOR * dt * 5.0

    def update_actor_physics(self, actor: AircraftActor, dt):
        if actor.crashed:
            return

        self.sanitize_actor(actor)

        target_roll_rate = actor.roll_input * ROLL_SENSITIVITY
        target_pitch_rate = actor.pitch_input * PITCH_SENSITIVITY
        target_yaw_rate = actor.yaw_input * YAW_SENSITIVITY

        actor.roll_rate += (target_roll_rate - actor.roll_rate) * CONTROL_RESPONSE * dt
        actor.pitch_rate += (target_pitch_rate - actor.pitch_rate) * CONTROL_RESPONSE * dt
        actor.yaw_rate += (target_yaw_rate - actor.yaw_rate) * CONTROL_RESPONSE * dt

        self.calculate_turn_coordination(actor, dt)

        # Coordinated turn adds yaw, but smoothly and with dt.
        actor.yaw_rate += actor.turn_rate * dt

        actor.roll += actor.roll_rate * dt
        actor.pitch += actor.pitch_rate * dt
        actor.yaw += actor.yaw_rate * dt

        actor.roll = clamp(wrap_angle_deg(actor.roll), -MAX_ROLL, MAX_ROLL)
        actor.pitch = clamp(actor.pitch, -90, 90)
        actor.yaw = actor.yaw % 360

        # Speed control is independent of turning.
        self.update_speed_control(actor, dt)

        forward_dir = actor.get_forward_vector()

        # Align current velocity toward nose direction gradually.
        if vec_length_sq(actor.velocity) < 1e-8:
            current_dir = forward_dir
        else:
            current_dir = actor.velocity.normalized()

        blend = clamp(DIR_BLEND_RATE * dt, 0.0, 1.0)
        new_dir = (current_dir * (1.0 - blend)) + (forward_dir * blend)
        if vec_length_sq(new_dir) < 1e-8:
            new_dir = forward_dir
        else:
            new_dir.normalize()

        # Small climb/dive influence from pitch so altitude still changes naturally.
        pitch_factor = clamp(actor.pitch / 45.0, -1.0, 1.0)
        vertical_bias = Vec3(0, 0, pitch_factor * 0.12)
        new_dir = new_dir + vertical_bias
        if vec_length_sq(new_dir) < 1e-8:
            new_dir = forward_dir
        else:
            new_dir.normalize()

        actor.velocity = new_dir * actor.current_speed
        actor.position += actor.velocity * dt

        actor.airspeed = actor.velocity.length()
        actor.altitude = max(0.0, actor.position.z)

        # G-force estimate
        bank_factor = abs(actor.roll) / max(1.0, MAX_ROLL)
        pitch_factor_g = abs(actor.pitch) / 90.0
        actor.g_force = clamp(1.0 + bank_factor * 4.0 + pitch_factor_g * 1.5, 0.1, MAX_G_FORCE)

        self.sanitize_actor(actor)

    # ========================================================
    # Enemy AI
    # ========================================================
    def update_enemy_ai(self, dt):
        e = self.enemy
        if e.crashed:
            return

        e.ai_update_timer -= dt

        if e.ai_update_timer <= 0 or (e.ai_waypoint - e.position).length() < ENEMY_WAYPOINT_REACH_DIST:
            e.ai_update_timer = self.rng.uniform(ENEMY_AI_UPDATE_TIME_MIN, ENEMY_AI_UPDATE_TIME_MAX)
            unit_offset = random_point_in_sphere(1.0, self.rng)
            waypoint_offset = Vec3(
                unit_offset.x * ENEMY_PATROL_RADIUS,
                unit_offset.y * ENEMY_PATROL_RADIUS,
                unit_offset.z * ENEMY_PATROL_VERTICAL_RADIUS,
            )
            e.ai_waypoint = e.ai_patrol_center + waypoint_offset
            e.ai_waypoint.z = clamp(
                e.ai_waypoint.z,
                ENEMY_PATROL_MIN_ALTITUDE,
                ENEMY_PATROL_MAX_ALTITUDE,
            )

        to_waypoint = e.ai_waypoint - e.position
        if vec_length_sq(to_waypoint) < 1e-8:
            to_waypoint = Vec3(0, 1, 0)

        desired_dir = to_waypoint.normalized()
        desired_heading = math.degrees(math.atan2(desired_dir.x, desired_dir.y)) % 360
        horizontal_len = math.sqrt(desired_dir.x ** 2 + desired_dir.y ** 2)
        desired_pitch = math.degrees(math.atan2(desired_dir.z, max(1e-6, horizontal_len)))

        heading_error = shortest_angle_delta(desired_heading, e.yaw)
        pitch_error = desired_pitch - e.pitch

        e.yaw_input = clamp(heading_error / 45.0, -1.0, 1.0)
        e.pitch_input = clamp(pitch_error / 25.0, -1.0, 1.0)
        e.roll_input = clamp(-heading_error / 60.0, -1.0, 1.0)

        dist = to_waypoint.length()
        if dist > 1800:
            e.target_speed = 220.0
        elif dist > 900:
            e.target_speed = 190.0
        else:
            e.target_speed = 160.0

        # Gentle variation, not a jump.
        if self.rng.random() < 0.008:
            e.target_speed += self.rng.uniform(-10, 10)

        e.target_speed = clamp(e.target_speed, ENEMY_MIN_SPEED, ENEMY_MAX_SPEED)
        if e.current_speed < e.target_speed - 5:
            e.speed_input = 1.0
        elif e.current_speed > e.target_speed + 5:
            e.speed_input = -1.0
        else:
            e.speed_input = 0.0

    # ========================================================
    # Boundary and game over
    # ========================================================
    def check_boundary(self):
        rel = self.player.position - self.enemy.position
        if rel.length() > CHASE_BOUNDARY_RADIUS and not self.game_over:
            self.game_over = True
            self.game_over_message = "MISSION FAILED: YOU LEFT THE CHASE BOUNDARY"
            self.player.crashed = True
            self.enemy.crashed = True
            self.player.velocity = Vec3(0, 0, 0)
            self.enemy.velocity = Vec3(0, 0, 0)
            self.player.roll_rate = self.player.pitch_rate = self.player.yaw_rate = 0.0
            self.enemy.roll_rate = self.enemy.pitch_rate = self.enemy.yaw_rate = 0.0
            self.player.turn_rate = self.enemy.turn_rate = 0.0
            print(self.game_over_message)

    # ========================================================
    # Camera and HUD
    # ========================================================
    def update_camera(self):
        if getattr(self, "headless", False):
            return
        if self.game_over and not self.player_camera_locked:
            camera_pos = self.player.position + Vec3(0, -70, 30)
            self.camera.setPos(camera_pos)
            self.camera.lookAt(self.player.position)
            return

        if self.camera_mode == 0 or self.player_camera_locked:
            forward = self.player.get_forward_vector()
            up = self.player.get_up_vector()
            camera_offset_world = forward * self.camera_offset.y + up * self.camera_offset.z
            camera_pos = self.player.position + camera_offset_world
            look_at = self.player.position + forward * self.camera_look_offset.y
            self.camera.setPos(camera_pos)
            self.camera.lookAt(look_at)
        else:
            camera_pos = self.player.position + Vec3(0, 0, 2)
            look_dir = self.player.get_forward_vector()
            self.camera.setPos(camera_pos)
            self.camera.lookAt(camera_pos + look_dir * 100)

    def update_hud(self):
        self.update_radar()
        if self.headless:
            return
        rel = self.player.position - self.enemy.position
        radial_distance = rel.length()
        radial_margin = CHASE_BOUNDARY_RADIUS - radial_distance

        x_clearance = CHASE_BOUNDARY_RADIUS - abs(rel.x)
        y_clearance = CHASE_BOUNDARY_RADIUS - abs(rel.y)
        z_clearance = CHASE_BOUNDARY_RADIUS - abs(rel.z)

        if self.game_over:
            hud_text = f"""
{self.game_over_message}

PLAYER POSITION:
  X: {self.player.position.x:.0f} m
  Y: {self.player.position.y:.0f} m
  Z: {self.player.position.z:.0f} m

DISTANCE TO ENEMY CENTER:
  X: {rel.x:+.0f} m
  Y: {rel.y:+.0f} m
  Z: {rel.z:+.0f} m
  RADIAL: {radial_distance:.0f} m
"""
        else:
            try:
                velocity_dir = self.player.velocity.normalized()
                forward_dir = self.player.get_forward_vector()
                dot = clamp(velocity_dir.dot(forward_dir), -1.0, 1.0)
                aoa = math.degrees(math.acos(dot))
                if velocity_dir.z < forward_dir.z:
                    aoa = -aoa
            except Exception:
                aoa = self.player.pitch

            missile_status = "READY" if (self.missile is not None and not self.missile.isHidden()) else "FIRED"

            hud_text = f"""
PLAYER SPEED: {self.player.current_speed:.0f} m/s
ENEMY SPEED: {self.enemy.current_speed:.0f} m/s

PLAYER ALTITUDE: {self.player.altitude:.0f} m
G-FORCE: {self.player.g_force:.1f} G
AoA: {aoa:.1f}°

PITCH: {self.player.pitch:.1f}°
ROLL: {self.player.roll:.1f}°
HEADING: {self.player.yaw:.1f}°
TURN RATE: {self.player.turn_rate:.1f}°/s
MAX ROLL: {MAX_ROLL:.1f}°

CHASE BOUNDARY:
  RADIAL DISTANCE TO ENEMY: {radial_distance:.0f} m
  RADIAL CLEARANCE: {radial_margin:.0f} m
  X CLEARANCE: {x_clearance:.0f} m
  Y CLEARANCE: {y_clearance:.0f} m
  Z CLEARANCE: {z_clearance:.0f} m

ENEMY OFFSET:
  X: {rel.x:+.0f} m
  Y: {rel.y:+.0f} m
  Z: {rel.z:+.0f} m

MISSILE: {missile_status}

CONTROLS:
  Arrow Keys - Pitch/Roll
  Q/E - Yaw
  W/S - Speed
  T - Engine Toggle
  C - Camera Mode
  M - Fire Missile
"""
        self.hud_text.setText(hud_text)

    # ========================================================
    # Update loop
    # ========================================================
    def update_jet_position(self, actor: AircraftActor):
        actor.sync_graphics()

    def update_radar(self):
        """Snapshot numerical entities only; destroyed aircraft leave the radar."""
        contacts = []
        for identity, team, actor in (("player", "friendly", self.player),
                                      ("enemy", "enemy", self.enemy)):
            if not actor.crashed and not (identity == "enemy" and self.target_destroyed):
                contacts.append(RadarContact(identity, team, "aircraft", tuple(actor.position)))
        for state in self.active_missiles:
            if "missile_id" in state and "position" in state:
                contacts.append(RadarContact(state["missile_id"], state["team"],
                                             "missile", tuple(state["position"])))
        self.radar_contacts = project_contacts(contacts, tuple(self.player.position), self.radar_config)
        if self.radar is not None:
            self.radar.update(self.radar_contacts)

    def update_task(self, task):
        dt = globalClock.getDt()
        if dt <= 0 or self.is_nan_value(dt):
            return Task.cont

        self._physics_accumulator += dt

        try:
            # Rendering may run faster or slower; physics always consumes 50 ms.
            while self._physics_accumulator >= 0.05 - 1e-12:
                self._physics_accumulator = max(0.0, self._physics_accumulator - 0.05)
                target_previous_position = Vec3(self.enemy.position)
                player_previous_position = Vec3(self.player.position)
                if not self.game_over:
                    self.update_enemy_ai(0.05)
                    self.update_actor_physics(self.player, 0.05)
                    self.update_actor_physics(self.enemy, 0.05)
                    self.check_boundary()
                self.update_jet_position(self.player)
                self.update_jet_position(self.enemy)
                self.update_missiles(0.05, target_previous_position, player_previous_position)
                if self.missile_vfx is not None:
                    self.missile_vfx.update_realtime(0.05)

            if self.enemy.boundary_sphere is not None:
                self.enemy.boundary_sphere.setPos(0, 0, 0)
                self.enemy.boundary_sphere.setScale(CHASE_BOUNDARY_RADIUS)

            self.update_camera()
            self.update_hud()

        except Exception as e:
            print(f"Error in update loop: {e}")
            self.player.position = Vec3(0, 0, 1000)
            self.player.velocity = Vec3(0, 150, 0)
            self.player.roll = 0.0
            self.player.pitch = 0.0
            self.player.yaw = 0.0
            self.player.current_speed = 150.0
            self.player.target_speed = 150.0
            self.player.crashed = False

        return Task.cont


if __name__ == "__main__":
    app = JetSim(enable_missile_vfx=True)
    app.run()
