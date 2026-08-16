"""DBZ-style visual effects rendered with OpenCV, additive-blended onto frames."""
import random
import math
import numpy as np
import cv2


def _glow_layer(shape):
    return np.zeros(shape, dtype=np.float32)


def blend_additive(frame, glow):
    """Additively blend a float32 BGR glow layer onto a uint8 frame."""
    out = frame.astype(np.float32) + glow
    return np.clip(out, 0, 255).astype(np.uint8)


class AuraParticles:
    """Rising spark particles around a point, used while an orb is charging."""

    def __init__(self):
        self.particles = []  # each: [x, y, vx, vy, life, max_life, color]

    def emit(self, cx, cy, radius, n=6, color=(255, 255, 255)):
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.3, 1.0) * radius
            x = cx + math.cos(ang) * r
            y = cy + math.sin(ang) * r
            vx = random.uniform(-0.6, 0.6)
            vy = random.uniform(-2.5, -0.8)
            life = random.uniform(15, 30)
            self.particles.append([x, y, vx, vy, life, life, color])

    def update_and_draw(self, glow):
        alive = []
        for p in self.particles:
            x, y, vx, vy, life, max_life, color = p
            x += vx
            y += vy
            life -= 1
            if life > 0:
                t = life / max_life
                rad = max(1, int(3 * t))
                c = tuple(int(ch * t) for ch in color)
                cv2.circle(glow, (int(x), int(y)), rad, c, -1, lineType=cv2.LINE_AA)
                alive.append([x, y, vx, vy, life, max_life, color])
        self.particles = alive


def draw_soft_glow_circle(glow, center, radius, color, intensity=1.0, rings=4):
    """Simulate a soft glow by drawing several alpha-decreasing rings."""
    cx, cy = int(center[0]), int(center[1])
    for i in range(rings, 0, -1):
        r = int(radius * i / rings)
        alpha = intensity * (1 - i / (rings + 1)) * 0.9
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(glow, (cx, cy), max(1, r), c, -1, lineType=cv2.LINE_AA)


class KiBlast:
    """A projectile energy orb fired from the palm in a direction."""

    def __init__(self, x, y, dx, dy, color, speed=22, radius=16, owner=None):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.color = color
        self.speed = speed
        self.radius = radius
        self.life = 60
        self.owner = owner

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.life -= 1

    def offscreen(self, w, h):
        return self.life <= 0 or self.x < -50 or self.x > w + 50 or self.y < -50 or self.y > h + 50

    def draw(self, glow):
        draw_soft_glow_circle(glow, (self.x, self.y), self.radius, self.color, intensity=1.2, rings=5)
        core = tuple(min(255, int(c * 1.5)) for c in self.color)
        cv2.circle(glow, (int(self.x), int(self.y)), max(2, self.radius // 3), core, -1, lineType=cv2.LINE_AA)
