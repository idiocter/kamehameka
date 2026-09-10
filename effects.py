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

# Color ramps per jutsu type - lower indices = deeper/bluer, higher =icier/whiter
RASENGAN_COLOR_RAMP = [
    (0.0, (255, 30, 10)),    # deep core - very blue
    (0.5, (255, 130, 40)),   # middle - standard rasengan
    (1.0, (255, 220, 200)),  # near-white rim
]
RASENSHURIKEN_COLOR_RAMP = [
    (0.0, (255, 30, 10)),    # deep core
    (0.5, (255, 180, 100)),  # middle - awakening
    (1.0, (255, 250, 250)),  # near-white fully charged
]

SOFT_SIGMA = 6.0    # bloom blur
SHARP_SIGMA = 1.2   # just enough to take the aliasing off an edge

# The chakra hierarchy, densest first. Position 0 is the compressed core, 1 the
# fastest-moving outer filaments. White is a luminosity, not a substance - it
# only ever appears at the far end of the ramp, on the thinnest structures.
CHAKRA_STOPS = [
    (0.00, (255, 55, 0)),      # deep electric blue - densest chakra
    (0.30, (255, 120, 35)),    # bright royal blue - inner rotational layer
    (0.55, (255, 185, 95)),    # cyan - wind release beginning to expand
    (0.80, (255, 225, 165)),   # pale icy blue - expanding wind chakra
    (1.00, (255, 250, 240)),   # near-white - highest apparent luminosity
]


def chakra_color(t):
    """Sample the chakra ramp; t=0 is the dense core, t=1 the outer highlights."""
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(CHAKRA_STOPS, CHAKRA_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(a + (b - a) * f for a, b in zip(c0, c1))
    return CHAKRA_STOPS[-1][1]


def jutsu_color(charge_frac, ramp):
    """Sample a jutsu-specific color ramp based on charge fraction.
    charge_frac: 0.0 .. 1.0
    ramp: list of (threshold, BGR color) sorted ascending
    """
    t = min(1.0, max(0.0, charge_frac))
    for (t0, c0), (t1, c1) in zip(ramp, ramp[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(a + (b - a) * f for a, b in zip(c0, c1))
    return ramp[-1][1]


_CHAKRA_LUT = np.array([chakra_color(i / 255.0) for i in range(256)], dtype=np.float32)


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
    """Two accumulation buffers: heavy-blurred bloom, and light-blurred detail.

    Buffers are reused across frames to avoid allocation overhead.
    """

    def __init__(self, shape):
        self.soft = _glow_layer(shape)
        self.sharp = _glow_layer(shape)
        self._shape = shape
        self._soft_blur = np.empty(shape, dtype=np.float32)
        self._sharp_blur = np.empty(shape, dtype=np.float32)

    def reset(self):
        """Clear buffers for next frame."""
        self.soft.fill(0.0)
        self.sharp.fill(0.0)

    def composite(self, frame, soft_sigma=SOFT_SIGMA, sharp_sigma=SHARP_SIGMA):
        # In-place blur to preallocated buffers (avoids allocation)
        cv2.GaussianBlur(self.soft, (0, 0), sigmaX=soft_sigma, sigmaY=soft_sigma, dst=self._soft_blur)
        cv2.GaussianBlur(self.sharp, (0, 0), sigmaX=sharp_sigma, sigmaY=sharp_sigma, dst=self._sharp_blur)
        # Combine and add to frame
        combined = self._soft_blur + self._sharp_blur
        out = frame.astype(np.float32) + combined
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8)


_SPHERE_CACHE = {}


def _sphere_patch(radius):
    """A dense sphere with deep-blue core grading to pale icy blue + 3D shading data.

    Returns dict with: 'color' (BGR float32), 'alpha', 'normal_z', 'depth', 'specular_mask'
    All cached per integer radius.
    """
    r = max(3, int(radius))
    cached = _SPHERE_CACHE.get(r)
    if cached is not None:
        return cached

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.hypot(xx, yy) / r                       # 0 at centre, 1 at rim
    valid = d <= 1.0

    # --- Base color from chakra ramp (core to rim) ---
    color_idx = np.clip(d ** 1.5 * 0.85 * 255, 0, 255).astype(np.uint8)
    base_color = _CHAKRA_LUT[color_idx]            # (H, W, 3) float32

    # --- Alpha: solid core, sharp falloff at edge ---
    alpha = np.clip((1.08 - d) / 0.22, 0.0, 1.0)
    density = 0.62 + 0.38 * np.clip(1.0 - d, 0.0, 1.0)
    alpha *= density

    # --- 3D normals (sphere surface) ---
    # For a sphere: normal = (x, y, z) / r where z = sqrt(r^2 - x^2 - y^2)
    z = np.sqrt(np.maximum(0.0, 1.0 - d * d))
    normal_z = z                                     # facing camera = 1.0 at center

    # --- Specular mask (Phong-like highlight) ---
    # Light from upper-left: light_dir ≈ (-0.3, -0.3, 0.9)
    # View dir = (0, 0, 1). Half-vector H = normalize(light + view)
    # Specular ~ max(0, N·H)^shininess
    shininess = 64.0
    hx, hy, hz = -0.21, -0.21, 0.98  # Pre-normalized half-vector
    nh = normal_z * hz  # nx*hx + ny*hy + nz*hz, but nx,ny small near center
    specular = np.power(np.maximum(0.0, nh), shininess)
    specular = np.where(valid, specular, 0.0)

    # --- Rim lighting (fresnel-like) ---
    rim = np.power(1.0 - normal_z, 2.5)
    rim = np.where(valid, rim, 0.0)

    # --- Depth for occlusion sorting ---
    depth = 1.0 - d  # 1 at center, 0 at rim

    patch = {
        'color': base_color,
        'alpha': alpha[..., None],
        'normal_z': normal_z[..., None],
        'depth': depth[..., None],
        'specular': specular[..., None],
        'rim': rim[..., None],
        'valid': valid[..., None],
        'radius': r,
    }
    _SPHERE_CACHE[r] = patch
    return patch


def _add_patch(dst, patch, cx, cy):
    """Add a BGR patch centred at (cx, cy), clipped to the destination bounds.
    Handles both legacy array patches and new dict patches with 3D shading.
    """
    if isinstance(patch, dict):
        color = patch['color']
        alpha = patch['alpha']
        specular = patch.get('specular')
        rim = patch.get('rim')
        valid = patch['valid']
        ph, pw = color.shape[:2]
    else:
        color = patch
        alpha = None
        specular = None
        rim = None
        valid = None
        ph, pw = patch.shape[:2]

    y0, x0 = int(cy) - ph // 2, int(cx) - pw // 2
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = min(dst.shape[0], y0 + ph), min(dst.shape[1], x0 + pw)
    if dy0 >= dy1 or dx0 >= dx1:
        return

    sy0, sx0 = dy0 - y0, dx0 - x0
    sy1, sx1 = sy0 + (dy1 - dy0), sx0 + (dx1 - dx0)

    if alpha is not None:
        a = alpha[sy0:sy1, sx0:sx1]
        dst[dy0:dy1, dx0:dx1] += color[sy0:sy1, sx0:sx1] * a
        if specular is not None:
            s = specular[sy0:sy1, sx0:sx1]
            dst[dy0:dy1, dx0:dx1] += s * 255.0  # White specular highlight
        if rim is not None:
            rm = rim[sy0:sy1, sx0:sx1]
            dst[dy0:dy1, dx0:dx1] += rm * 80.0  # Rim light (cool blue-white)
    else:
        dst[dy0:dy1, dx0:dx1] += color[sy0:sy1, sx0:sx1]


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


def draw_rasengan(layers, cx, cy, radius, frame_idx, charge_frac=1.0):
    """Blue chakra sphere with interwoven currents projected over its surface."""
    cx, cy = int(cx), int(cy)
    charge = min(1.0, max(0.0, charge_frac))
    r = max(4.0, radius * (1.0 + 0.018 * math.sin(frame_idx * 0.23)))
    brightness = 0.65 + 0.35 * charge
    draw_soft_glow_circle(layers.soft, (cx, cy), r * 1.4,
                          (255, 135, 45), intensity=0.85 * brightness)

    # Scale only this contribution; never multiply other effects in its bounds.
    sphere = _sphere_patch(r)
    body = (sphere['color'] * sphere['alpha'] * 0.65
            + sphere['rim'] * np.array((90, 65, 25), dtype=np.float32))
    _add_patch(layers.sharp, body * brightness, cx, cy)

    # Different tilted great-circle currents give the orb volume. Broken arcs
    # and variable speeds keep the texture flowing rather than looking metallic.
    phase = frame_idx * 0.085
    for i in range(16):
        tilt = i * 2.39996
        start = phase * (1.0 + (i % 4) * 0.19) + i * 1.7
        theta = np.linspace(start, start + 3.8, 48)
        depth = np.sin(theta)
        orbit = r * (0.58 + 0.025 * i)
        x = orbit * np.cos(theta)
        y = orbit * depth * (0.24 + 0.035 * (i % 7))
        points = np.column_stack((cx + x * math.cos(tilt) - y * math.sin(tilt),
                                  cy + x * math.sin(tilt) + y * math.cos(tilt)))
        for j in range(0, len(points) - 1, 8):
            front = 0.5 + 0.5 * float(depth[j])
            color = _scale((255, 175 + 55 * front, 90 + 100 * front),
                           brightness * (0.45 + 0.45 * front))
            cv2.polylines(layers.sharp, [points[j:j + 9].astype(np.int32)],
                          False, color, max(1, int(r * 0.025)), cv2.LINE_AA)

    # Fine outer contour and small curling streams read clearly at palm size.
    cv2.circle(layers.sharp, (cx, cy), int(r),
               _scale((255, 210, 130), 0.7 * brightness), 1, cv2.LINE_AA)
    for i in range(3):
        theta = np.linspace(0, 4.8, 45)
        spiral_r = r * (0.08 + 0.13 * theta)
        angle = theta - phase * 1.7 + i * math.tau / 3
        points = np.column_stack((cx + spiral_r * np.cos(angle),
                                  cy + spiral_r * np.sin(angle) * 0.8))
        cv2.polylines(layers.sharp, [points.astype(np.int32)], False,
                      _scale((255, 225, 165), 0.85 * brightness), 1, cv2.LINE_AA)


BLADE_SWEEP = -0.16   # radians the centreline curls back over the blade's length
# Peak half-width as a fraction of blade length. Roughly half the angular space
# has to stay empty or the four petals merge into a flower and the gaps - which
# are what make it read as a shuriken - disappear.
BLADE_WIDTH = 0.18
_WIDTH_FRONT = 0.18   # exponents of the width profile: t**FRONT * (1-t)**BACK
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


FILAMENTS_PER_BLADE = 38
_BLADE_SAMPLES = 15


def _filament_layout(count=FILAMENTS_PER_BLADE, seed=17):
    """Fixed per-strand parameters: (lateral seat, start t, twist rate, phase, brightness).

    Generated once and reused every frame. Re-rolling these per frame would make
    the blade boil like static instead of holding a shape that spins.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        s = rng.uniform(-1.0, 1.0)
        # Bias seats toward the blade's axis. The twist term below is sinusoidal,
        # so strands already linger near their extremes; uniform seating on top
        # of that piles them onto the two edges and the blade reads as two rails
        # with a hollow middle.
        seat = math.copysign(abs(s) ** 1.6, s)
        out.append((
            seat,
            rng.uniform(0.0, 0.30),      # how far out the strand starts
            rng.uniform(0.6, 2.4),       # how fast it twists around the blade axis
            rng.uniform(0, 2 * math.pi),
            rng.uniform(0.55, 1.0),      # brightness
        ))
    return out


_FILAMENTS = _filament_layout()


def draw_rasenshuriken(layers, cx, cy, radius, frame_idx, emergence=1.0,
                       halo=RASENSHURIKEN_HALO, velocity=None, speed=0.0,
                       thrown=False, throw_frame=0, energize_frame=0):
    """Four tapered white wind blades surrounding a swirling blue chakra core.

    Short angular afterimages suggest speed while preserving the four-point
    silhouette. Throw growth is restrained so the core and tips stay readable.
    """
    cx, cy = int(cx), int(cy)
    e = min(1.0, max(0.0, emergence))
    if e == 0.0:
        draw_rasengan(layers, cx, cy, radius, frame_idx)
        return
    expansion = 1.0 + (0.25 * min(1.0, max(0, throw_frame) / 12.0) if thrown else 0.0)
    core_radius = max(4.0, radius * expansion)
    outer = core_radius * (1.0 + 3.2 * e)
    inner = core_radius * 0.78
    # Integrating a bounded spin-up avoids the phase jumps of age * spin_rate.
    age = max(0, energize_frame)
    spin = frame_idx * 0.16 + 0.012 * min(age, 14) ** 2
    if thrown:
        spin += throw_frame * 0.12
    if velocity is not None and speed > 0:
        spin += 0.08 * math.atan2(velocity[1], velocity[0])

    for k in range(4):
        angle = spin + k * math.pi / 2
        # Trailing exposures belong in bloom, leaving dark gaps between points.
        for lag, strength in ((0.18, 0.10), (0.09, 0.18)):
            trail = _blade_spine(cx, cy, angle - lag, inner, outer, _BLADE_SAMPLES)
            cv2.fillPoly(layers.soft, [_blade_polygon(trail, 1.1)],
                         _scale(halo, strength * e), cv2.LINE_AA)
        spine = _blade_spine(cx, cy, angle, inner, outer, _BLADE_SAMPLES)
        cv2.fillPoly(layers.sharp, [_blade_polygon(spine)],
                     _scale((235, 213, 175), 0.72 * e), cv2.LINE_AA)
        cv2.fillPoly(layers.sharp, [_blade_polygon(spine, 0.7)],
                     _scale(RASENSHURIKEN_COLOR, 0.90 * e), cv2.LINE_AA)
        # Flowing hairline streaks inside the luminous wind envelope.
        for seat, t_start, twist, phase, bright in _FILAMENTS[::3]:
            pts = []
            for i, (x, y, tx, ty, width) in enumerate(spine):
                t = i / _BLADE_SAMPLES
                if t < t_start:
                    continue
                offset = seat * math.cos(phase + twist * t - frame_idx * 0.22)
                pts.append((x + tx * width * offset, y + ty * width * offset))
            cv2.polylines(layers.sharp, [np.asarray(pts, dtype=np.int32)], False,
                          _scale(RASENSHURIKEN_COLOR, e * (0.78 + 0.22 * bright)),
                          1, cv2.LINE_AA)

    # Broken wind arcs around the hub, not a full circular outer blade border.
    for i in range(3):
        start = math.degrees(-frame_idx * 0.12) + i * 120
        cv2.ellipse(layers.sharp, (cx, cy),
                    (max(2, int(core_radius * 1.5)), max(2, int(core_radius * 0.65))),
                    18, start, start + 85, _scale(RASENSHURIKEN_COLOR, 0.8 * e),
                    max(1, int(core_radius * 0.04)), cv2.LINE_AA)
    draw_rasengan(layers, cx, cy, core_radius, frame_idx)


class JutsuBlast:
    """A thrown Rasengan or Rasenshuriken travelling in a straight line."""

    def __init__(self, x, y, dx, dy, style, speed=28, radius=26, velocity=None):
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
        self.velocity = velocity or (dx, dy)  # normalized or raw velocity

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.life -= 1
        self.frame += 1

    def offscreen(self, w, h):
        return self.life <= 0 or self.x < -50 or self.x > w + 50 or self.y < -50 or self.y > h + 50

    def draw(self, layers):
        if self.style == "rasenshuriken":
            speed_factor = math.hypot(self.dx, self.dy) * self.speed
            # Use the same wind silhouette in flight as in the hand.
            draw_rasenshuriken(layers, self.x, self.y, self.radius, self.frame,
                               velocity=self.velocity, speed=speed_factor,
                               thrown=True, throw_frame=self.frame)
        else:
            draw_rasengan(layers, self.x, self.y, self.radius, self.frame)


class WindDome:
    """Rasenshuriken detonation - an expanding shell of wind blades tearing outward."""

    def __init__(self, x, y, max_radius=260, life=28, color=RASENSHURIKEN_COLOR):
        self.x, self.y = int(x), int(y)
        self.max_radius = max_radius
        self.life = life
        self.max_life = life
        self.color = color
        self.needles = [(random.uniform(0, 2 * math.pi), random.uniform(0.7, 1.15)) for _ in range(26)]
        self._finished = False

    def update(self):
        self.life -= 1
        if self.life <= 0:
            self._finished = True

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

    @property
    def finished(self):
        return self._finished


class WindSplash:
    """Impact splash that appears after WindDome completes - radiating wind blades."""

    def __init__(self, x, y, max_radius=300, life=20, color=RASENSHURIKEN_COLOR):
        self.x, self.y = int(x), int(y)
        self.max_radius = max_radius
        self.life = life
        self.max_life = life
        self.color = color
        self.needles = [(random.uniform(0, 2 * math.pi), random.uniform(0.8, 1.3)) for _ in range(32)]

    def update(self):
        self.life -= 1

    def offscreen(self, w, h):
        return self.life <= 0

    def draw(self, layers):
        t = 1.0 - self.life / self.max_life           # 1 -> 0 over the splash
        radius = self.max_radius * t
        fade = (self.life / self.max_life) ** 0.7

        # Radiating blades
        for ang, span in self.needles:
            r0 = radius * 0.3 * span
            r1 = radius * 0.9 * span
            cv2.line(layers.sharp,
                     (int(self.x + math.cos(ang) * r0), int(self.y + math.sin(ang) * r0)),
                     (int(self.x + math.cos(ang) * r1), int(self.y + math.sin(ang) * r1)),
                     _scale(self.color, fade), max(1, int(2 * fade)), lineType=cv2.LINE_AA)

        # Outer glowing ring
        cv2.circle(layers.sharp, (self.x, self.y), max(2, int(radius)),
                   _scale(self.color, fade * 0.6), max(1, int(4 * fade)), lineType=cv2.LINE_AA)

        # Inner core
        cv2.circle(layers.sharp, (self.x, self.y), max(2, int(radius * 0.4)),
                   _scale(self.color, fade), max(1, int(2 * fade)), lineType=cv2.LINE_AA)

        # Soft bloom underneath
        draw_soft_glow_circle(layers.soft, (self.x, self.y), radius * 0.6, self.color,
                              intensity=fade * 1.5, rings=5)
