"""Jutsu visual effects rendered with OpenCV, additive-blended onto frames.

Effects draw onto GlowLayers, a pair of float32 BGR accumulation buffers. The
`soft` buffer is blurred hard for bloom; the `sharp` buffer is barely blurred so
shapes with real silhouettes - notably the Rasenshuriken's blades - keep their
edges. Both are added onto the camera frame at the end, so overlapping effects
accumulate into light rather than painting over each other.
"""
import random
import math
import numpy as np
import cv2

RASENGAN_COLOR = (255, 130, 40)        # BGR - saturated chakra blue
RASENSHURIKEN_COLOR = (255, 250, 245)  # BGR - white, with only a whisper of blue
RASENSHURIKEN_HALO = (255, 165, 85)    # BGR - blue, for bloom and rim light only

SOFT_SIGMA = 6.0    # bloom blur
SHARP_SIGMA = 1.2   # just enough to take the aliasing off an edge


def _glow_layer(shape):
    return np.zeros(shape, dtype=np.float32)


def blend_additive(frame, glow):
    """Additively blend a float32 BGR glow layer onto a uint8 frame."""
    out = frame.astype(np.float32) + glow
    return np.clip(out, 0, 255).astype(np.uint8)


def _scale(color, k):
    """Brighten a colour without hue drift.

    Clamping each channel independently would pin the blue channel at 255 first
    and let green keep climbing, washing chakra blue out to cyan - so cap the
    multiplier instead and slide the colour up its own hue line.
    """
    peak = max(color)
    if peak > 0:
        k = min(k, 255.0 / peak)
    return tuple(float(ch) * k for ch in color)


class GlowLayers:
    """Two accumulation buffers: heavy-blurred bloom, and light-blurred detail."""

    def __init__(self, shape):
        self.soft = _glow_layer(shape)
        self.sharp = _glow_layer(shape)

    def composite(self, frame, soft_sigma=SOFT_SIGMA, sharp_sigma=SHARP_SIGMA):
        soft = cv2.GaussianBlur(self.soft, (0, 0), sigmaX=soft_sigma, sigmaY=soft_sigma)
        sharp = cv2.GaussianBlur(self.sharp, (0, 0), sigmaX=sharp_sigma, sigmaY=sharp_sigma)
        return blend_additive(frame, soft + sharp)


def draw_soft_glow_circle(dst, center, radius, color, intensity=1.0, rings=4):
    """Simulate a soft glow by drawing several alpha-decreasing rings."""
    cx, cy = int(center[0]), int(center[1])
    for i in range(rings, 0, -1):
        r = int(radius * i / rings)
        alpha = intensity * (1 - i / (rings + 1)) * 0.9
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(dst, (cx, cy), max(1, r), c, -1, lineType=cv2.LINE_AA)


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

    def update_and_draw(self, layers):
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
                cv2.circle(layers.soft, (int(x), int(y)), rad, c, -1, lineType=cv2.LINE_AA)
                alive.append([x, y, vx, vy, life, max_life, color])
        self.particles = alive


def draw_rasengan(layers, cx, cy, radius, frame_idx, charge_frac=1.0, color=RASENGAN_COLOR):
    """A dense sphere of chakra with shell layers spiralling around a bright core."""
    cx, cy = int(cx), int(cy)
    pulse = 1.0 + 0.06 * math.sin(frame_idx * 0.45)
    r = max(4.0, radius * pulse)
    brightness = 0.75 + 0.35 * charge_frac

    draw_soft_glow_circle(layers.soft, (cx, cy), r * 1.35, color,
                          intensity=1.15 * brightness, rings=6)

    # Rotating shell arcs at differing radii and speeds read as a spiral swirl.
    for i in range(3):
        rr = max(2, int(r * (0.55 + 0.2 * i)))
        tilt = (frame_idx * (9.0 + 4.0 * i) + i * 55) % 360
        sweep_start = (frame_idx * (13.0 - 3.0 * i)) % 360
        cv2.ellipse(layers.sharp, (cx, cy), (rr, max(2, int(rr * 0.62))), tilt,
                    sweep_start, sweep_start + 250, _scale(color, 1.15 * brightness),
                    max(1, int(r * 0.09)), lineType=cv2.LINE_AA)

    # Wisps skimming the surface.
    for i in range(4):
        a = frame_idx * 0.32 + i * (math.pi / 2)
        wx = cx + math.cos(a) * r * 1.05
        wy = cy + math.sin(a) * r * 0.75
        cv2.circle(layers.sharp, (int(wx), int(wy)), max(1, int(r * 0.1)),
                   _scale(color, 1.3 * brightness), -1, lineType=cv2.LINE_AA)

    # A small, pale-blue core rather than a big white one - a white disc this
    # close to the centre saturates the whole orb and kills the blue read.
    core = _scale((255, 215, 180), brightness)
    cv2.circle(layers.sharp, (cx, cy), max(2, int(r * 0.22)), core, -1, lineType=cv2.LINE_AA)


BLADE_SWEEP = -0.40   # radians the centreline curls back over the blade's length
# Peak half-width as a fraction of blade length. Roughly half the angular space
# has to stay empty or the four petals merge into a flower and the gaps - which
# are what make it read as a shuriken - disappear.
BLADE_WIDTH = 0.17
_WIDTH_FRONT = 0.45   # exponents of the width profile: t**FRONT * (1-t)**BACK
_WIDTH_BACK = 1.0     # higher = finer point at the tip
# Peak of t**a * (1-t)**b sits at t = a/(a+b); divide through by its value there
# so BLADE_WIDTH means what it says.
_WIDTH_PEAK_T = _WIDTH_FRONT / (_WIDTH_FRONT + _WIDTH_BACK)
_WIDTH_NORM = _WIDTH_PEAK_T ** _WIDTH_FRONT * (1 - _WIDTH_PEAK_T) ** _WIDTH_BACK


def _blade_spine(cx, cy, angle, inner, outer, samples):
    """Sample the blade's curved centreline: (x, y, tangential unit vector, width)."""
    span = outer - inner
    out = []
    for i in range(samples + 1):
        t = i / samples
        r = inner + span * t
        th = angle + BLADE_SWEEP * t   # the curl - a straight blade reads as a plus sign
        w = BLADE_WIDTH * outer * (t ** _WIDTH_FRONT) * ((1 - t) ** _WIDTH_BACK) / _WIDTH_NORM
        out.append((cx + math.cos(th) * r, cy + math.sin(th) * r, -math.sin(th), math.cos(th), w))
    return out


def _blade_polygon(spine, scale=1.0):
    """Outline of one petal: out along one edge of the spine, back along the other."""
    left = [(x + tx * w * scale, y + ty * w * scale) for x, y, tx, ty, w in spine]
    right = [(x - tx * w * scale, y - ty * w * scale) for x, y, tx, ty, w in spine]
    pts = left + right[::-1]
    return np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)


def _blade_striation(spine, offset):
    """A line running the length of the blade, `offset` across its half-width."""
    pts = [(x + tx * w * offset, y + ty * w * offset) for x, y, tx, ty, w in spine]
    return np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)


def draw_rasenshuriken(layers, cx, cy, radius, frame_idx,
                       color=RASENSHURIKEN_COLOR, halo=RASENSHURIKEN_HALO):
    """The Rasengan core wrapped in a spinning four-bladed wind shuriken.

    Each blade is traced along a swept centreline rather than built from corners:
    it pinches at the hub, bellies out early, then tapers to a curled point. Four
    straight-sided slivers give you the right symmetry but read as a throwing-star
    pictogram; the curl and the belly are what make it look like compressed wind.

    Blades go on the sharp layer - at bloom-blur strength the notches between them
    wash out and the whole thing reads as a disc instead of a star.
    """
    cx, cy = int(cx), int(cy)
    # ~5 deg/frame. A 4-blade star repeats every 90 deg, so this is one visual
    # cycle every ~18 frames - fast enough to feel like a spin, slow enough to
    # actually see the shape.
    spin = frame_idx * 0.09
    blade_len = radius * 4.5
    inner = radius * 0.55

    for k in range(4):
        spine = _blade_spine(cx, cy, spin + k * (math.pi / 2), inner, blade_len, samples=16)
        poly = _blade_polygon(spine)

        # Blue bloom hugs the blade's edge rather than flooding its interior -
        # a filled blue underlay would tint the whole body and lose the white.
        cv2.polylines(layers.soft, [_blade_polygon(spine, scale=1.2)], True,
                      _scale(halo, 0.6), 7, lineType=cv2.LINE_AA)

        # Translucent body inside a crisp rim - cel-shaded, not a solid slab.
        # fillPoly, not fillConvexPoly: a curved petal is not convex.
        cv2.fillPoly(layers.sharp, [poly], _scale(color, 0.52), lineType=cv2.LINE_AA)

        for offset in (-0.55, 0.0, 0.55):
            cv2.polylines(layers.sharp, [_blade_striation(spine, offset)], False,
                          _scale(color, 0.62), 1, lineType=cv2.LINE_AA)

        cv2.polylines(layers.sharp, [poly], True, _scale(color, 1.0), 2, lineType=cv2.LINE_AA)
        cv2.polylines(layers.soft, [poly], True, _scale(halo, 0.75), 5, lineType=cv2.LINE_AA)

    # Faint disc the blades sweep through.
    cv2.circle(layers.sharp, (cx, cy), max(2, int(blade_len * 0.99)),
               _scale(halo, 0.22), 1, lineType=cv2.LINE_AA)

    draw_rasengan(layers, cx, cy, radius, frame_idx, charge_frac=1.0, color=color)


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

    def draw(self, layers):
        if self.style == "rasenshuriken":
            draw_rasenshuriken(layers, self.x, self.y, self.radius, self.frame)
        else:
            draw_rasengan(layers, self.x, self.y, self.radius, self.frame * 3)


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

    def draw(self, layers):
        t = 1.0 - self.life / self.max_life           # 0 -> 1 over the blast
        radius = self.max_radius * (1 - (1 - t) ** 2)  # bursts out fast, then eases
        fade = (self.life / self.max_life) ** 1.5

        cv2.circle(layers.sharp, (self.x, self.y), max(2, int(radius)), _scale(self.color, fade),
                   max(1, int(6 * fade)), lineType=cv2.LINE_AA)
        cv2.circle(layers.sharp, (self.x, self.y), max(1, int(radius * 0.72)),
                   _scale(self.color, fade * 0.45), max(1, int(3 * fade)), lineType=cv2.LINE_AA)

        for ang, span in self.needles:
            r0 = radius * 0.55 * span
            r1 = radius * 1.08 * span
            cv2.line(layers.sharp,
                     (int(self.x + math.cos(ang) * r0), int(self.y + math.sin(ang) * r0)),
                     (int(self.x + math.cos(ang) * r1), int(self.y + math.sin(ang) * r1)),
                     _scale(self.color, fade * 0.9), max(1, int(2 * fade)), lineType=cv2.LINE_AA)

        draw_soft_glow_circle(layers.soft, (self.x, self.y), radius * 0.4, self.color,
                              intensity=fade * 1.2, rings=4)
