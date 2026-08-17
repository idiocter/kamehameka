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


def draw_charge_orb(glow, cx, cy, radius, color, pulse_t):
    """Pulsing glowing orb used while an energy ball is forming."""
    pulse = 1.0 + 0.15 * math.sin(pulse_t * 0.4)
    draw_soft_glow_circle(glow, (cx, cy), radius * pulse, color, intensity=1.3, rings=6)
    cv2.circle(glow, (int(cx), int(cy)), max(2, int(radius * 0.25)), (255, 255, 255), -1, lineType=cv2.LINE_AA)


def draw_lightning_bolt(glow, x1, y1, x2, y2, color, thickness=2, segments=6, jitter=14):
    """A jagged bolt from (x1, y1) to (x2, y2), rendered as a displaced zigzag."""
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) + 1e-6
    perp = (-dy / n, dx / n)

    points = [(x1, y1)]
    for i in range(1, segments):
        t = i / segments
        bx = x1 + dx * t
        by = y1 + dy * t
        edge_fade = min(t, 1 - t) * 2  # taper jitter toward both ends
        offset = random.uniform(-jitter, jitter) * edge_fade
        points.append((bx + perp[0] * offset, by + perp[1] * offset))
    points.append((x2, y2))

    pts = [(int(px), int(py)) for px, py in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(glow, a, b, color, thickness, lineType=cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        cv2.line(glow, a, b, (255, 255, 255), max(1, thickness // 2), lineType=cv2.LINE_AA)


def draw_charging_lightning(glow, cx, cy, orb_radius, charge_frac, color, frame_idx):
    """Lightning arcs crackling inward from surrounding space, feeding a charging orb."""
    n_arcs = 1 + int(charge_frac * 5)
    reach = orb_radius * (2.5 + charge_frac * 2.0)

    for i in range(n_arcs):
        if (frame_idx + i * 7) % 3 != 0:  # flicker - not every arc fires every frame
            continue
        ang = random.uniform(0, 2 * math.pi)
        sx = cx + math.cos(ang) * reach
        sy = cy + math.sin(ang) * reach
        draw_lightning_bolt(glow, sx, sy, cx, cy, color, thickness=2, segments=5, jitter=10)
