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
# Collision capture radius. This exceeds the maximum ~41.75 m missile travel in
# one capped 0.05 s physics update at maximum launch velocity.
MISSILE_HIT_RADIUS = 50.0

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

# Enemy AI
ENEMY_PATROL_RADIUS = 250.0
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
    def __init__(self, enable_missile_vfx=False):
        self.rng = random.Random()
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
        if not self.game_over:
            self.camera_mode = (self.camera_mode + 1) % 2

    # ========================================================
    # Missile
    # ========================================================
    def enable_missile_effects(self, cinematic=True, slowmo=8.0):
        from missile_vfx import MissileVFX

        if self.missile_vfx is not None:
            self.missile_vfx.destroy()
        self.missile_vfx = MissileVFX(self, cinematic=cinematic, slowmo=slowmo)
        return self.missile_vfx

    def destroy(self):
        effects = getattr(self, "missile_vfx", None)
        if effects is not None:
            effects.destroy()
            self.missile_vfx = None
        super().destroy()

    def attach_missile(self):
        try:
            self.missile = self.loader.loadModel(Filename.fromOsSpecific(asset_path(MISSILE_MODEL_PATH)))
        except Exception as e:
            print("Error loading missile model:", e)
            self.missile = None
            return

        self.missile.reparentTo(self.player.missile_mount_node)

        try:
            ts_list = self.missile.findAllTextureStages()
            for ts in ts_list:
                self.missile.setTexture(ts, None)
        except Exception:
            pass

        try:
            missile_mat = Material()
            missile_mat.setDiffuse((0.62, 0.62, 0.62, 1.0))
            missile_mat.setAmbient((0.62, 0.62, 0.62, 1.0))
            missile_mat.setSpecular((0.10, 0.10, 0.10, 1.0))
            missile_mat.setShininess(6.0)
            self.missile.setMaterial(missile_mat, 1)
            self.missile.setColorScale(1, 1, 1, 1)
            self.missile.setTwoSided(True)
        except Exception:
            pass

        self.missile.setScale(MISSILE_RELATIVE_SCALE)
        self.missile.setPos(0, 0, 0)
        self.missile.setHpr(0, 0, 0)

    def fire_missile(self):
        if self.game_over:
            return
        if self.missile is None:
            print("Missile model is not available.")
            return
        if self.missile.isHidden():
            return

        launch_pos = self.missile.getPos(self.render)
        launch_hpr = self.missile.getHpr(self.render)

        projectile = self.loader.loadModel(Filename.fromOsSpecific(asset_path(MISSILE_MODEL_PATH)))
        projectile.reparentTo(self.render)
        projectile.setPos(launch_pos)
        projectile.setHpr(launch_hpr)

        world_scale = self.player.visual_root.getScale(self.render).x * MISSILE_RELATIVE_SCALE
        projectile.setScale(world_scale)

        try:
            ts_list = projectile.findAllTextureStages()
            for ts in ts_list:
                projectile.setTexture(ts, None)
        except Exception:
            pass

        try:
            missile_mat = Material()
            missile_mat.setDiffuse((0.62, 0.62, 0.62, 1.0))
            missile_mat.setAmbient((0.62, 0.62, 0.62, 1.0))
            missile_mat.setSpecular((0.10, 0.10, 0.10, 1.0))
            missile_mat.setShininess(6.0)
            projectile.setMaterial(missile_mat, 1)
            projectile.setColorScale(1, 1, 1, 1)
            projectile.setTwoSided(True)
        except Exception:
            pass

        q = Quat()
        q.setHpr(launch_hpr)
        launch_direction = q.xform(Vec3(0, 0, 1))
        if vec_length_sq(launch_direction) < 1e-8:
            launch_direction = Vec3(0, 0, 1)
        else:
            launch_direction.normalize()

        projectile_state = {
            "node": projectile,
            "velocity": launch_direction * MISSILE_LAUNCH_SPEED + self.player.velocity,
            "age": 0.0,
            "spawn_pos": Vec3(launch_pos),
            "hit_registered": False,
        }
        self.active_missiles.append(projectile_state)
        if self.missile_vfx is not None:
            self.missile_vfx.on_launch(projectile_state)
        self.missile.hide()
        print("Missile launched.")

    def register_missile_hit(self, missile_state):
        """Register one projectile hit and expose the event for future consumers."""
        if missile_state.get("hit_registered", False):
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

    def update_missiles(self, dt):
        if not self.active_missiles:
            return

        kept = []
        for m in self.active_missiles:
            node = m["node"]
            m["age"] += dt
            node.setPos(node.getPos() + m["velocity"] * dt)
            if self.missile_vfx is not None:
                self.missile_vfx.on_motion(m)

            if (node.getPos() - self.enemy.position).length() <= MISSILE_HIT_RADIUS:
                self.register_missile_hit(m)
                self._remove_projectile(m, hit=True)
                continue

            if m["age"] > MISSILE_MAX_LIFETIME:
                self._remove_projectile(m)
                continue

            if (node.getPos() - m["spawn_pos"]).length() > MISSILE_MAX_RANGE:
                self._remove_projectile(m)
                continue

            if node.getZ() < -1000:
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
            waypoint_offset = random_point_in_sphere(ENEMY_PATROL_RADIUS, self.rng)
            waypoint_offset.z *= 0.60
            e.ai_waypoint = e.ai_patrol_center + waypoint_offset

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
        if self.game_over:
            camera_pos = self.player.position + Vec3(0, -70, 30)
            self.camera.setPos(camera_pos)
            self.camera.lookAt(self.player.position)
            return

        if self.camera_mode == 0:
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

    def update_task(self, task):
        dt = globalClock.getDt()
        if dt <= 0 or self.is_nan_value(dt):
            return Task.cont

        dt = min(dt, 0.05)

        try:
            if not self.game_over:
                self.update_enemy_ai(dt)
                self.update_actor_physics(self.player, dt)
                self.update_actor_physics(self.enemy, dt)
                self.check_boundary()

            self.update_jet_position(self.player)
            self.update_jet_position(self.enemy)

            if self.enemy.boundary_sphere is not None:
                self.enemy.boundary_sphere.setPos(0, 0, 0)
                self.enemy.boundary_sphere.setScale(CHASE_BOUNDARY_RADIUS)

            self.update_missiles(dt)
            self.update_camera()
            self.update_hud()
            if self.missile_vfx is not None:
                self.missile_vfx.update_realtime(dt)

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
