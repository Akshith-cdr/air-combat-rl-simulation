"""Shared live-only wrapper: retain the terminal scene for impact animation."""

import gymnasium as gym
import numpy as np
from panda3d.core import Vec3


class PhysicsFrameBuffer:
    """Interpolate scene transforms between fixed steps; never mutate actors.

    The caller restores scene/camera transforms after every rendered frame.
    Radar is deliberately outside this buffer and keeps the newest physics state.
    """

    def __init__(self, sim):
        self.sim = sim
        self.previous = []
        self.current = []

    def before_step(self):
        self.previous = [(actor.root, Vec3(actor.position), actor.root.getQuat())
                         for actor in (self.sim.player, self.sim.enemy)]
        simple = getattr(self.sim, "simple_missile_visuals", None)
        if simple is not None:
            simple.before_step(self.sim.active_missiles)

    def after_step(self):
        self.current = [(actor.root, Vec3(actor.position), actor.root.getQuat(), actor.root.getTransform())
                        for actor in (self.sim.player, self.sim.enemy)]
        simple = getattr(self.sim, "simple_missile_visuals", None)
        if simple is not None:
            simple.sync(self.sim.active_missiles)

    def apply(self, alpha):
        for start, end in zip(self.previous, self.current):
            node, position, quat = start
            _, end_position, end_quat, _ = end
            node.setPos(position * (1 - alpha) + end_position * alpha)
            if quat.dot(end_quat) < 0:
                end_quat = -end_quat
            rotation = quat * (1 - alpha) + end_quat * alpha
            rotation.normalize()
            node.setQuat(rotation)
        # Follow the interpolated player scene root, with unchanged chase offsets.
        player = self.sim.player.root
        rotation, position = player.getQuat(), player.getPos()
        forward, up = rotation.xform(Vec3(0, 1, 0)), rotation.xform(Vec3(0, 0, 1))
        self.sim.camera.setPos(position + forward * self.sim.camera_offset.y + up * self.sim.camera_offset.z)
        self.sim.camera.lookAt(position + forward * self.sim.camera_look_offset.y)
        simple = getattr(self.sim, "simple_missile_visuals", None)
        if simple is not None:
            simple.draw(alpha, view=(self.sim.camera.getPos(self.sim.render),
                                    self.sim.camLens.getHfov(), self.sim.win.getXSize()))

    def restore(self):
        for node, _, _, transform in self.current:
            node.setTransform(transform)
        self.sim.update_camera()


class PreserveTerminalVisuals(gym.Wrapper):
    """Prevent DummyVecEnv's auto-reset from erasing the confirmed hit scene."""

    def __init__(self, env):
        super().__init__(env)
        self._terminal_observation = None

    def reset(self, *, seed=None, options=None):
        if self._terminal_observation is not None and seed is None and options is None:
            observation = self._terminal_observation
            self._terminal_observation = None
            return observation.copy(), {"terminal_visuals_preserved": True}
        self._terminal_observation = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        result = self.env.step(action)
        observation, _, terminated, truncated, _ = result
        if terminated or truncated:
            self._terminal_observation = np.asarray(observation, dtype=np.float32).copy()
        return result
