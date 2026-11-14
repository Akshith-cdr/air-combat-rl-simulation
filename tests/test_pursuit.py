"""Checks for the pursuit/ pipeline.  Run:  python -m unittest tests.test_pursuit -v"""

import csv
import math
import unittest
from pathlib import Path

import numpy as np

from pursuit import config as C
from pursuit.baselines import PursuitController
from pursuit.env import PursuitEnv, band_excess, segment_min_distance
from pursuit.scenarios import Scenario, reviewed_six_suite, sweep_suite

ROOT = Path(__file__).resolve().parents[1]


def scenario(**overrides):
    values = dict(scenario_id="T", seed=1, player_heading=0.0, player_speed=200.0, altitude=1500.0,
                  range_m=3000.0, bearing=0.0, delta_altitude=0.0, target_relative_heading=0.0,
                  target_speed=190.0)
    values.update(overrides)
    return Scenario(**values)


def run(env, controller, s):
    obs, _ = env.reset(options={"scenario": s})
    while True:
        obs, _, terminated, truncated, info = env.step(controller.act(env, obs))
        if terminated or truncated:
            return info


class Geometry(unittest.TestCase):
    def test_band_excess(self):
        self.assertEqual(band_excess(750), 0)
        self.assertEqual(band_excess(C.BAND_MIN_M), 0)
        self.assertEqual(band_excess(C.BAND_MAX_M), 0)
        self.assertEqual(band_excess(1200), 200)
        self.assertEqual(band_excess(300), 200)

    def test_swept_collision_distance_catches_pass_through(self):
        # Relative position goes from 15 m ahead to 15 m behind within one step.
        self.assertAlmostEqual(segment_min_distance([0, 15, 3], [0, -15, 3]), 3.0)
        self.assertAlmostEqual(segment_min_distance([0, 30, 0], [0, 20, 0]), 20.0)


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.env = PursuitEnv()

    def tearDown(self):
        self.env.close()

    def test_bearing_and_heading_conventions(self):
        for heading, bearing in ((0, 90), (90, 0), (45, -120), (300, 180)):
            obs, _ = self.env.reset(options={"scenario": scenario(player_heading=heading, bearing=bearing,
                                                                  target_relative_heading=90)})
            self.assertAlmostEqual(obs[10], 3000.0, places=2)
            self.assertAlmostEqual((obs[15] - bearing + 180) % 360 - 180, 0.0, places=3)  # bearing error
            enemy, player = self.env.sim.enemy, self.env.sim.player
            # Velocity and nose agree for both aircraft.
            for actor in (player, enemy):
                forward, velocity = actor.get_forward_vector(), actor.velocity.normalized()
                self.assertGreater(forward.dot(velocity), 0.9999)
            target_heading = math.degrees(math.atan2(enemy.velocity.x, enemy.velocity.y)) % 360
            self.assertAlmostEqual((target_heading - (heading + 90) + 180) % 360 - 180, 0.0, places=3)

    def test_reviewed_six_matches_previous_evaluation(self):
        old = ROOT / "results/final_eval_1000_stage7/final_eval_1000.csv"
        with open(old) as handle:
            previous = [(r["scenario_id"], int(r["scenario_seed"])) for r in csv.DictReader(handle)]
        current = [(s.scenario_id, s.seed) for s in reviewed_six_suite()]
        self.assertEqual(current, previous)

    def test_sweep_covers_all_bearings(self):
        suite = sweep_suite()
        self.assertEqual(len(suite), 12 * 4 * 3 * 7)
        self.assertEqual(sorted({s.bearing_sector for s in suite}), list(range(0, 360, 30)))
        self.assertEqual(len({(s.scenario_id, s.seed) for s in suite}), len(suite))


class Episodes(unittest.TestCase):
    def setUp(self):
        self.env = PursuitEnv(record_trajectory=True)

    def tearDown(self):
        self.env.close()

    def test_launch_only_inside_gate(self):
        controller = PursuitController()
        for bearing in (0, 90, 180, -90):
            info = run(self.env, controller, scenario(bearing=bearing))
            summary = info["episode_summary"]
            self.assertTrue(summary["launched"])
            self.assertGreaterEqual(summary["launch_range_m"], C.BAND_MIN_M)
            self.assertLessEqual(summary["launch_range_m"], C.BAND_MAX_M)
            self.assertGreater(summary["launch_cos_ata"], C.COS_ATA_GATE)

    def test_no_launch_when_never_aligned(self):
        # Fly straight while a target 800 m to the right crosses: inside the band, never aligned.
        obs, _ = self.env.reset(options={"scenario": scenario(range_m=800, bearing=90,
                                                              target_relative_heading=0, target_speed=200)})
        for _ in range(40):
            obs, _, terminated, truncated, info = self.env.step(np.zeros(4))
            self.assertFalse(info["launched"])

    def test_collision_ends_episode_as_failure(self):
        # Head-on, 400 m apart, both straight and level: closes at 390 m/s.
        self.env.reset(options={"scenario": scenario(range_m=400, target_relative_heading=180)})
        self.env.sim.update_enemy_ai = lambda dt: None      # target flies straight (inputs stay 0)
        for _ in range(200):
            _, reward, terminated, truncated, info = self.env.step(np.zeros(4))
            if terminated or truncated:
                break
        self.assertEqual(info["episode_summary"]["outcome"], "collision")
        self.assertFalse(info["episode_summary"]["success"])
        self.assertLess(info["episode_summary"]["min_distance_m"], C.COLLISION_RADIUS_M)

    def test_trajectory_recorded(self):
        info = run(self.env, PursuitController(), scenario())
        self.assertEqual(len(self.env.trajectory), info["episode_summary"]["steps"] + 1)
        self.assertIn("m_x", self.env.trajectory[-1])

    def test_lead_pursuit_aims_at_target_inside_band(self):
        lead = PursuitController(lead=True)
        p, vp, vt = np.zeros(3), np.array([0.0, 200.0, 0.0]), np.array([190.0, 0.0, 0.0])
        far, near = np.array([0.0, 3000.0, 0.0]), np.array([0.0, 900.0, 0.0])
        self.assertFalse(np.allclose(lead.aim_point(p, vp, far, vt), far))   # leads when far
        np.testing.assert_allclose(lead.aim_point(p, vp, near, vt), near)    # pure inside band

    def test_target_patrol_steers_to_waypoint(self):
        # A fixed waypoint 1 km to the target's left must be approached, not circled.
        self.env.reset(options={"scenario": scenario()})
        enemy = self.env.sim.enemy
        self.assertTrue(self.env.sim.correct_enemy_heading)
        heading = math.radians(-enemy.yaw)                     # compass heading
        left = np.array([-math.cos(heading), math.sin(heading), 0.0])
        waypoint = np.array(enemy.position) + 1000.0 * left
        enemy.ai_waypoint = type(enemy.position)(*waypoint)
        enemy.ai_update_timer = 1e9                            # no new waypoint
        start = np.linalg.norm(waypoint - np.array(enemy.position))
        for _ in range(100):                                   # 5 s
            self.env.sim.update_enemy_ai(C.SIMULATION_DT)
            self.env.sim.update_actor_physics(enemy, C.SIMULATION_DT)
        self.assertLess(np.linalg.norm(waypoint - np.array(enemy.position)), 0.6 * start)


if __name__ == "__main__":
    unittest.main()
