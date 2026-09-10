"""Headless regression checks for motion, rendering, and effect lifecycles."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

import effects as FX
import gestures as G


class EffectsTests(unittest.TestCase):
    def setUp(self):
        self.layers = FX.GlowLayers((240, 320, 3))

    def test_unformed_wind_is_rasengan(self):
        FX.draw_rasengan(self.layers, 160, 120, 24, 19.5)
        expected = self.layers.sharp.copy()
        self.layers.reset()
        FX.draw_rasenshuriken(self.layers, 160, 120, 24, 19.5, emergence=0)
        np.testing.assert_array_equal(expected, self.layers.sharp)

    def test_render_clips_at_edges_and_supports_fractional_time(self):
        for center in ((0, 0), (319, 239), (-150, 120), (470, 120)):
            for emergence in (0, 0.1, 0.5, 1):
                self.layers.reset()
                FX.draw_rasenshuriken(self.layers, *center, 32, 12.5, emergence=emergence)
                output = self.layers.composite(np.full((240, 320, 3), 80, np.uint8))
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(np.isfinite(self.layers.sharp).all())

    def test_spin_keeps_its_speed_after_formation(self):
        phases = [FX.shuriken_phase(t, t) for t in (13.99, 14, 14.01, 15)]
        self.assertAlmostEqual((phases[1] - phases[0]) / 0.01,
                               (phases[2] - phases[1]) / 0.01, places=3)
        self.assertAlmostEqual(phases[3] - phases[1], 0.384)

    def test_throw_preserves_starting_phase_and_emergence(self):
        blast = FX.JutsuBlast(160, 120, 1, 0, 'rasenshuriken',
                              animation_frame=47.5, energize_frame=6, emergence=0.4)
        self.assertEqual(blast.start_phase, FX.shuriken_phase(47.5, 6))
        FX.draw_rasenshuriken(self.layers, 160, 120, 26, 47.5, emergence=0.4,
                              energize_frame=6)
        expected = self.layers.sharp.copy()
        self.layers.reset()
        blast.draw(self.layers)
        np.testing.assert_array_equal(expected, self.layers.sharp)

    def test_travel_is_independent_of_update_rate(self):
        slow = FX.JutsuBlast(30, 50, 3, 4, 'rasengan', speed=20)
        fast = FX.JutsuBlast(30, 50, 3, 4, 'rasengan', speed=20)
        for _ in range(10):
            slow.update(1)
        for _ in range(20):
            fast.update(0.5)
        self.assertAlmostEqual(slow.x, fast.x)
        self.assertAlmostEqual(slow.y, fast.y)
        self.assertAlmostEqual(math.hypot(slow.x - 30, slow.y - 50), 200)

    def test_edge_throw_detonates_before_leaving_camera(self):
        for x, y, dx, dy in ((300, 120, 1, 0), (10, 120, -1, 0),
                              (160, 220, 0, 1), (160, 10, 0, -1), (160, 120, 1, 1)):
            effects = [FX.JutsuBlast(x, y, dx, dy, 'rasenshuriken', bounds=(320, 240))]
            for _ in range(20):
                self.layers.reset()
                effects = FX.advance_projectiles(effects, self.layers, 320, 240)
                if isinstance(effects[0], FX.WindDome):
                    break
            self.assertIsInstance(effects[0], FX.WindDome)
            self.assertTrue(0 <= effects[0].x < 320)
            self.assertTrue(0 <= effects[0].y < 240)

    def test_dome_then_splash_then_cleanup(self):
        effects = [FX.WindDome(160, 120, life=2)]
        effects = FX.advance_projectiles(effects, self.layers, 320, 240)
        self.assertIsInstance(effects[0], FX.WindDome)
        effects = FX.advance_projectiles(effects, self.layers, 320, 240)
        self.assertEqual(len(effects), 1)
        self.assertIsInstance(effects[0], FX.WindSplash)
        for _ in range(20):
            self.layers.reset()
            effects = FX.advance_projectiles(effects, self.layers, 320, 240)
        self.assertEqual(effects, [])

    def test_expired_burst_draws_no_light(self):
        for cls in (FX.WindDome, FX.WindSplash):
            effect = cls(160, 120, life=2)
            effect.update(3)
            effect.draw(self.layers)
        self.assertFalse(self.layers.soft.any())
        self.assertFalse(self.layers.sharp.any())

    def test_throw_direction_uses_pixel_aspect_ratio(self):
        dx, dy, speed = G.throw_motion((0.6, 0.6), (0.5, 0.5), 640, 480)
        self.assertAlmostEqual(dx, 0.8)
        self.assertAlmostEqual(dy, 0.6)
        self.assertAlmostEqual(speed, 0.125)
        _, _, half_step_speed = G.throw_motion((0.55, 0.55), (0.5, 0.5), 640, 480, 0.5)
        self.assertAlmostEqual(speed, half_step_speed)
        self.assertEqual(G.throw_motion((0.5, 0.5), None, 640, 480), (0, 0, 0))

    def test_palm_size_uses_camera_dimensions(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        landmarks[G.MIDDLE_MCP] = SimpleNamespace(x=0.6, y=0.6)
        self.assertAlmostEqual(G.palm_size_pixels(landmarks, 640, 480), 80)


if __name__ == '__main__':
    unittest.main()
