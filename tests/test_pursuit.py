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

    def test_shaping_cannot_be_farmed_by_a_loop(self):
        # Any closed loop of (range, cos ATA) states earns exactly zero shaping,
        # however fast or slow each leg is (this is what clipping used to break).
        from pursuit.env import potential, reward_terms
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(3, 40))
            loop = list(zip(rng.uniform(300, 6000, n), rng.uniform(-1, 1, n)))
            loop.append(loop[0])
            total = sum(reward_terms(d0, d1, c0, c1, 200, 1500, 0)["shaping"]
                        for (d0, c0), (d1, c1) in zip(loop, loop[1:]))
            self.assertAlmostEqual(total, 0.0, places=9)
        # Holding a state earns nothing; the potential is highest aligned inside the band.
        self.assertEqual(reward_terms(750, 750, 0.3, 0.3, 200, 1500, 0)["shaping"], 0.0)
        best = potential(750, 1.0)
        self.assertAlmostEqual(best, C.SHAPING_RANGE_WEIGHT + C.SHAPING_ALIGN_WEIGHT)
        for d in (300, 1200, 3000, 9000):
            self.assertLess(potential(d, 1.0), best)
        self.assertGreaterEqual(potential(1e6, -1.0), 0.0)

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

    def test_new_suites_sizes_altitudes_and_independent_seeds(self):
        from pursuit.scenarios import development_suite, sweep_alt_suite, tuning_suite
        dev, tune, alt = development_suite(), tuning_suite(), sweep_alt_suite()
        self.assertEqual((len(dev), len(tune), len(alt)), (72, 288, 864))
        self.assertEqual({s.delta_altitude for s in dev}, {-450.0})
        self.assertEqual({s.target_relative_heading for s in dev}, {90.0, 180.0, 270.0})
        self.assertEqual({s.delta_altitude for s in tune}, {-300.0, 0.0, 300.0})
        self.assertNotIn(0.0, {s.delta_altitude for s in alt})
        self.assertTrue(all(abs(s.delta_altitude) <= 500 for s in alt))
        # Test seeds are never used for development or tuning.
        test_seeds = {s.seed for s in sweep_suite() + alt + reviewed_six_suite()}
        self.assertFalse(test_seeds & {s.seed for s in dev + tune})
        # The original sweep is unchanged (same ids and seeds as the reviewed version).
        self.assertTrue(all(s.delta_altitude == 0.0 and s.scenario_id.startswith("SW-") for s in sweep_suite()))


class Reporting(unittest.TestCase):
    def test_holm_and_mcnemar(self):
        from pursuit.report import holm, mcnemar_p
        self.assertEqual(mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_p(0, 6), 2 / 64)
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_run_names(self):
        from pursuit.failures import split_run
        self.assertEqual(split_run("ppo_seed3_sweep_alt"), ("ppo", "sweep_alt"))
        self.assertEqual(split_run("pn_pursuit_reviewed_six"), ("pn_pursuit", "reviewed_six"))
        self.assertEqual(split_run("ppo_rule_speed_seed1_sweep"), ("ppo_rule_speed", "sweep"))

    def test_aspect_classes(self):
        from pursuit.evaluate import aspect_class
        self.assertEqual([aspect_class(a) for a in (0, 45, 46, 134, 135, 180)],
                         ["head_on", "head_on", "beam", "beam", "tail", "tail"])


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
        lead = PursuitController("lead")
        p, vp, vt = np.zeros(3), np.array([0.0, 200.0, 0.0]), np.array([190.0, 0.0, 0.0])
        far, near = np.array([0.0, 3000.0, 0.0]), np.array([0.0, 900.0, 0.0])
        self.assertFalse(np.allclose(lead.aim_point(p, vp, far, vt), far))   # leads when far
        np.testing.assert_allclose(lead.aim_point(p, vp, near, vt), near)    # pure inside band

    def test_pn_turns_towards_a_crossing_target_and_points_inside_band(self):
        pn = PursuitController("pn")
        p, vp = np.zeros(3), np.array([0.0, 200.0, 0.0])
        # Target 3 km ahead crossing left-to-right: the line of sight rotates clockwise -> turn right.
        heading_error, pitch = pn.pn_command(p, vp, np.array([0.0, 3000.0, 0.0]), np.array([190.0, 0, 0]), 0.0)
        self.assertGreater(heading_error, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        # Target climbing: the line of sight rotates up -> pitch up (relative to the nose, not the velocity).
        _, pitch = pn.pn_command(p, vp, np.array([0.0, 3000.0, 0.0]), np.array([0, 0, 50.0]), -10.0)
        self.assertGreater(pitch, -10.0)
        # Inside the band, or far off the nose, PN hands over to pure pursuit.
        self.assertIsNone(pn.pn_command(p, vp, np.array([0.0, 900.0, 0.0]), np.zeros(3), 0.0))
        self.assertIsNone(pn.pn_command(p, vp, np.array([3000.0, -100.0, 0.0]), np.zeros(3), 0.0))

    def test_pn_pursuit_launches(self):
        info = run(self.env, PursuitController("pn"), scenario(bearing=30, target_relative_heading=90))
        self.assertTrue(info["episode_summary"]["launched"])

    def test_speed_swap_mixes_steering_and_throttle(self):
        from pursuit.baselines import SpeedSwap

        class Fixed:
            def __init__(self, action):
                self.action = np.array(action, np.float32)

            def reset(self):
                pass

            def act(self, env, obs):
                return self.action

        swap = SpeedSwap(Fixed([0.1, 0.2, 0.3, 0.4]), Fixed([-1, -1, -1, -0.9]), "test")
        np.testing.assert_allclose(swap.act(None, None), [0.1, 0.2, 0.3, -0.9], rtol=1e-6)

    def test_separation_penalty_only_after_launch(self):
        from pursuit.env import separation_penalty
        self.assertEqual(separation_penalty(C.SEPARATION_RADIUS_M + 1), 0.0)
        self.assertAlmostEqual(separation_penalty(0.0), -1.0)
        self.assertAlmostEqual(separation_penalty(C.SEPARATION_RADIUS_M / 2), -0.5)
        # Before launch the term is zero even at close range.
        self.env.reset(options={"scenario": scenario(range_m=200, bearing=90, target_relative_heading=0)})
        _, _, _, _, info = self.env.step(np.zeros(4))
        self.assertFalse(info["launched"])
        self.assertEqual(info["reward_terms"]["separation"], 0.0)

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
