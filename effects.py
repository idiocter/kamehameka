"""Jutsu visual effects rendered with OpenCV, additive-blended onto frames.

Everything draws onto a float32 BGR "glow" layer which the caller blurs and adds
back over the camera frame, so overlapping effects accumulate into light rather
than painting over each other.
"""
import random
import math
import numpy as np
import cv2

RASENGAN_COLOR = (255, 170, 70)        # BGR - chakra blue
RASENSHURIKEN_COLOR = (255, 225, 170)  # BGR - paler, colder blue-white


def _glow_layer(shape):
    return np.zeros(shape, dtype=np.float32)


def blend_additive(frame, glow):
    """Additively blend a float32 BGR glow layer onto a uint8 frame."""
    out = frame.astype(np.float32) + glow
    return np.clip(out, 0, 255).astype(np.uint8)


def _scale(color, k):
    return tuple(min(255.0, float(ch) * k) for ch in color)


def draw_soft_glow_circle(glow, center, radius, color, intensity=1.0, rings=4):
    """Simulate a soft glow by drawing several alpha-decreasing rings."""
    cx, cy = int(center[0]), int(center[1])
    for i in range(rings, 0, -1):
        r = int(radius * i / rings)
        alpha = intensity * (1 - i / (rings + 1)) * 0.9
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(glow, (cx, cy), max(1, r), c, -1, lineType=cv2.LINE_AA)


class AuraParticles:
    """Rising spark particles around a point, used while a jutsu is charging."""

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


def draw_rasengan(glow, cx, cy, radius, frame_idx, charge_frac=1.0, color=RASENGAN_COLOR):
    """A dense sphere of chakra with shell layers spiralling around a white core."""
    cx, cy = int(cx), int(cy)
    pulse = 1.0 + 0.06 * math.sin(frame_idx * 0.45)
    r = max(4.0, radius * pulse)
    brightness = 0.7 + 0.5 * charge_frac

    draw_soft_glow_circle(glow, (cx, cy), r * 1.3, color, intensity=1.1 * brightness, rings=6)

    # Rotating shell arcs at differing radii and speeds read as a spiral swirl.
    for i in range(3):
        rr = max(2, int(r * (0.55 + 0.2 * i)))
        tilt = (frame_idx * (9.0 + 4.0 * i) + i * 55) % 360
        sweep_start = (frame_idx * (13.0 - 3.0 * i)) % 360
        cv2.ellipse(glow, (cx, cy), (rr, max(2, int(rr * 0.62))), tilt,
                    sweep_start, sweep_start + 250, _scale(color, 1.15 * brightness),
                    max(1, int(r * 0.09)), lineType=cv2.LINE_AA)

    # Wisps skimming the surface.
    for i in range(4):
        a = frame_idx * 0.32 + i * (math.pi / 2)
        wx = cx + math.cos(a) * r * 1.05
        wy = cy + math.sin(a) * r * 0.75
        cv2.circle(glow, (int(wx), int(wy)), max(1, int(r * 0.1)),
                   _scale(color, 1.3 * brightness), -1, lineType=cv2.LINE_AA)

    cv2.circle(glow, (cx, cy), max(2, int(r * 0.34)),
               _scale((255, 255, 255), brightness), -1, lineType=cv2.LINE_AA)


def draw_rasenshuriken(glow, cx, cy, radius, frame_idx, color=RASENSHURIKEN_COLOR):
    """The Rasengan core wrapped in a fast-spinning four-bladed wind disc."""
    cx, cy = int(cx), int(cy)
    spin = frame_idx * 0.34
    blade_len = radius * 2.6
    inner = radius * 0.85

    for k in range(4):
        a = spin + k * (math.pi / 2)
        # Swept-back petal: the leading edge reaches further than the trailing edge.
        pts = [
            (cx + math.cos(a - 0.42) * inner, cy + math.sin(a - 0.42) * inner),
            (cx + math.cos(a - 0.12) * blade_len, cy + math.sin(a - 0.12) * blade_len),
            (cx + math.cos(a + 0.05) * blade_len * 0.92, cy + math.sin(a + 0.05) * blade_len * 0.92),
            (cx + math.cos(a + 0.42) * inner, cy + math.sin(a + 0.42) * inner),
        ]
        poly = np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)
        cv2.fillConvexPoly(glow, poly, _scale(color, 0.42), lineType=cv2.LINE_AA)
        cv2.polylines(glow, [poly], True, _scale(color, 1.0), 1, lineType=cv2.LINE_AA)

    # Thin outer ring sells the motion blur of the disc.
    cv2.circle(glow, (cx, cy), max(2, int(blade_len * 0.98)), _scale(color, 0.28), 1, lineType=cv2.LINE_AA)

    draw_rasengan(glow, cx, cy, radius, frame_idx, charge_frac=1.0, color=color)


class JutsuBlast:
    """A thrown Rasengan or Rasenshuriken travelling in a straight line."""

    def __init__(self, x, y, dx, dy, style, speed=28, radius=26):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.style = style
        self.speed = speed
        self.radius = radius
        self.frame = 0
        self.detonates = style == "rasenshuriken"
        # The Rasenshuriken gets a short fuse so it goes off inside the frame
        # instead of sailing offscreen; a plain Rasengan just flies away.
        self.life = 16 if self.detonates else 60

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.life -= 1
        self.frame += 1

    def offscreen(self, w, h):
        return self.life <= 0 or self.x < -50 or self.x > w + 50 or self.y < -50 or self.y > h + 50

    def draw(self, glow):
        if self.style == "rasenshuriken":
            draw_rasenshuriken(glow, self.x, self.y, self.radius, self.frame * 3)
        else:
            draw_rasengan(glow, self.x, self.y, self.radius, self.frame * 3)


class WindDome:
    """Rasenshuriken detonation - an expanding shell of wind blades tearing outward."""

    def __init__(self, x, y, max_radius=260, life=28, color=RASENSHURIKEN_COLOR):
        self.x, self.y = int(x), int(y)
        self.max_radius = max_radius
        self.life = life
        self.max_life = life
        self.color = color
        self.needles = [(random.uniform(0, 2 * math.pi), random.uniform(0.7, 1.15)) for _ in range(26)]

    def update(self):
        self.life -= 1

    def offscreen(self, w, h):
        return self.life <= 0

    def draw(self, glow):
        t = 1.0 - self.life / self.max_life           # 0 -> 1 over the blast
        radius = self.max_radius * (1 - (1 - t) ** 2)  # bursts out fast, then eases
        fade = (self.life / self.max_life) ** 1.5

        cv2.circle(glow, (self.x, self.y), max(2, int(radius)), _scale(self.color, fade),
                   max(1, int(6 * fade)), lineType=cv2.LINE_AA)
        cv2.circle(glow, (self.x, self.y), max(1, int(radius * 0.72)), _scale(self.color, fade * 0.45),
                   max(1, int(3 * fade)), lineType=cv2.LINE_AA)

        for ang, span in self.needles:
            r0 = radius * 0.55 * span
            r1 = radius * 1.08 * span
            cv2.line(glow,
                     (int(self.x + math.cos(ang) * r0), int(self.y + math.sin(ang) * r0)),
                     (int(self.x + math.cos(ang) * r1), int(self.y + math.sin(ang) * r1)),
                     _scale(self.color, fade * 0.9), max(1, int(2 * fade)), lineType=cv2.LINE_AA)

        draw_soft_glow_circle(glow, (self.x, self.y), radius * 0.4, self.color, intensity=fade * 1.2, rings=4)
