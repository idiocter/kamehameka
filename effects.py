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
    """Two accumulation buffers: heavy-blurred bloom, and light-blurred detail."""

    def __init__(self, shape):
        self.soft = _glow_layer(shape)
        self.sharp = _glow_layer(shape)

    def composite(self, frame, soft_sigma=SOFT_SIGMA, sharp_sigma=SHARP_SIGMA):
        soft = cv2.GaussianBlur(self.soft, (0, 0), sigmaX=soft_sigma, sigmaY=soft_sigma)
        sharp = cv2.GaussianBlur(self.sharp, (0, 0), sigmaX=sharp_sigma, sigmaY=sharp_sigma)
        return blend_additive(frame, soft + sharp)


_SPHERE_CACHE = {}


def _sphere_patch(radius):
    """A dense sphere with a deep-blue core grading out to pale icy blue.

    Built as one array rather than stacked circles because the layers are added,
    not painted over - concentric additive circles would just pile up into a
    white blob at the centre, which is the opposite of the intended hierarchy.
    Cached per integer radius; the geometry only depends on size.
    """
    r = max(3, int(radius))
    patch = _SPHERE_CACHE.get(r)
    if patch is not None:
        return patch

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.hypot(xx, yy) / r                       # 0 at centre, 1 at the rim
    # Exponent holds the deep core wide and pushes the pale shades into a thin
    # outer shell, so the sphere reads as dense rather than as a soft gradient.
    color = _CHAKRA_LUT[np.clip(d ** 1.5 * 0.85 * 255, 0, 255).astype(np.uint8)]
    # Solid through the body, then a quick fade - a compressed sphere needs a
    # defined boundary, not a soft cloud.
    alpha = np.clip((1.08 - d) / 0.22, 0.0, 1.0)
    density = 0.62 + 0.38 * np.clip(1.0 - d, 0.0, 1.0)
    patch = color * (alpha * density)[..., None]

    _SPHERE_CACHE[r] = patch
    return patch


def _add_patch(dst, patch, cx, cy):
    """Add a BGR patch centred at (cx, cy), clipped to the destination bounds."""
    ph, pw = patch.shape[:2]
    y0, x0 = int(cy) - ph // 2, int(cx) - pw // 2
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = min(dst.shape[0], y0 + ph), min(dst.shape[1], x0 + pw)
    if dy0 >= dy1 or dx0 >= dx1:
        return
    dst[dy0:dy1, dx0:dx1] += patch[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]


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
    """A compact sphere of compressed chakra rotating in several directions at once.

    Deep electric blue at the core grading out through royal blue to a pale icy
    rim, with spiral currents wrapping around and through it on independent axes
    so the rotation reads as multidirectional rather than a single spin.
    """
    cx, cy = int(cx), int(cy)
    pulse = 1.0 + 0.05 * math.sin(frame_idx * 0.45)
    r = max(4.0, radius * pulse)
    brightness = 0.78 + 0.30 * charge_frac

    # Atmospheric glow, drawn as an annulus rather than stacked filled circles:
    # those all overlap at the centre and dump ~2.7x their colour there, washing
    # the deep-blue core out to white.
    cv2.circle(layers.soft, (cx, cy), max(2, int(r * 1.3)),
               _scale(chakra_color(0.32), 0.85 * brightness),
               max(2, int(r * 0.8)), lineType=cv2.LINE_AA)

    _add_patch(layers.sharp, _sphere_patch(r) * brightness, cx, cy)

    # Spiral currents on independent axes. Each ellipse is a great circle seen
    # edge-on at its own tilt, so together they read as one sphere turning
    # several ways at once instead of a stack of flat rings.
    for i in range(4):
        rr = max(2, int(r * (0.55 + 0.15 * i)))
        squash = 0.20 + 0.30 * abs(math.sin(frame_idx * (0.06 + 0.018 * i) + i))
        tilt = (frame_idx * (7.0 + 5.0 * i) + i * 47) % 360
        start = (frame_idx * (11.0 - 2.5 * i)) % 360
        cv2.ellipse(layers.sharp, (cx, cy), (rr, max(2, int(rr * squash))), tilt,
                    start, start + 230, _scale(chakra_color(0.62), 1.05 * brightness),
                    max(1, int(r * 0.07)), lineType=cv2.LINE_AA)

    # Near-white highlights ride the fastest-moving surface currents.
    for i in range(4):
        a = frame_idx * 0.30 + i * (math.pi / 2)
        wx = cx + math.cos(a) * r * 0.92
        wy = cy + math.sin(a) * r * 0.66
        cv2.circle(layers.sharp, (int(wx), int(wy)), max(1, int(r * 0.09)),
                   _scale(chakra_color(1.0), 0.9 * brightness), -1, lineType=cv2.LINE_AA)


BLADE_SWEEP = -0.40   # radians the centreline curls back over the blade's length
# Peak half-width as a fraction of blade length. Roughly half the angular space
# has to stay empty or the four petals merge into a flower and the gaps - which
# are what make it read as a shuriken - disappear.
BLADE_WIDTH = 0.20
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
                       halo=RASENSHURIKEN_HALO, velocity=None, speed=0.0):
    """A compact chakra core inside an enormous four-pointed wind shuriken.

    The four points are not solid blades. Each is a dense bundle of razor-thin
    filaments twisting around the blade axis, so the silhouette reads as sharp and
    aerodynamic while the substance stays fibrous and distributed - a solid fill
    looks like sheet metal, which is the one thing this technique is not.

    Colour runs outward along the chakra ramp: the core stays deep saturated blue
    while the fastest outer filaments carry the near-white highlights.

    `emergence` (0..1) drives the transformation - the wind chakra expands out of
    the sphere rather than the sphere simply being scaled up.

    `velocity` and `speed` influence filament twist: a fast throw winds the
    filaments into a more dynamic, spiralled appearance.
    """
    cx, cy = int(cx), int(cy)
    # ~5 deg/frame. A 4-blade star repeats every 90 deg, so this is one visual
    # cycle every ~18 frames - fast enough to feel like a spin, slow enough to
    # actually see the shape.
    spin = frame_idx * 0.09
    e = min(1.0, max(0.0, emergence))

    # --- velocity-dependent filament twist ---
    twist_offset = 0.0
    if velocity is not None and speed > 0.0:
        # Normalize velocity vector; faster speed = more twist
        vx, vy = velocity
        v_norm = math.hypot(vx, vy) + 1e-6
        # Map speed to a twist factor in [-0.2, 0.2]; sign gives handedness
        twist_factor = (vy / v_norm) * min(1.0, speed / 30.0)
        twist_offset = twist_factor * 0.15  # radians max offset

    blade_len = radius * (1.0 + 3.5 * e)
    # Blades start at the sphere's surface, not inside it - filaments crossing
    # the core would additively wash the deep blue out to white.
    inner = radius * 1.02
    chunk = max(1, _BLADE_SAMPLES // 3)

    for k in range(4):
        angle = spin + k * (math.pi / 2)
        spine = _blade_spine(cx, cy, angle, inner, blade_len, samples=_BLADE_SAMPLES)

        # Volumetric glow fills the blade's envelope, giving the bundle mass
        # without the filaments themselves ever becoming a solid surface.
        cv2.fillPoly(layers.soft, [_blade_polygon(spine, scale=1.1)],
                     _scale(halo, 0.22 * e), lineType=cv2.LINE_AA)

        for seat, t_start, twist, phase, bright in _FILAMENTS:
            pts = []
            for i, (x, y, tx, ty, w) in enumerate(spine):
                t = i / _BLADE_SAMPLES
                if t < t_start:
                    continue
                # Strands spiral around the blade axis, so they cross over one
                # another instead of lying in flat parallel stripes.
                off = seat * math.cos(phase + (twist + twist_offset) * t + frame_idx * 0.11)
                pts.append((x + tx * w * off, y + ty * w * off))
            if len(pts) < 3:
                continue
            arr = np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)

            # Drawn in radial chunks so colour climbs the ramp along the strand.
            for c0 in range(0, len(arr) - 1, chunk):
                seg = arr[c0:c0 + chunk + 1]
                if len(seg) < 2:
                    continue
                t_mid = t_start + (c0 + chunk * 0.5) / _BLADE_SAMPLES
                shade = chakra_color(min(1.0, 0.28 + t_mid * 0.85))
                cv2.polylines(layers.sharp, [seg], False,
                              _scale(shade, bright * (0.5 + 0.5 * e)), 1, lineType=cv2.LINE_AA)

        # Microscopic cutting structures: tiny slivers lying *along* the blade,
        # embedded inside the bundle. Anything angled across the blade reads as a
        # scratch laid over the top of it rather than wind inside it.
        for j, (seat, _t0, _tw, phase, _br) in enumerate(_FILAMENTS[::3]):
            i = 6 + (j * 3) % max(1, _BLADE_SAMPLES - 7)
            sx, sy, stx, sty, sw = spine[i]
            off = seat * math.cos(phase + frame_idx * 0.11)
            px, py = sx + stx * sw * off, sy + sty * sw * off
            # Radial direction is the tangent rotated a quarter turn.
            rx, ry = sty, -stx
            ln = sw * 0.55 * e
            cv2.line(layers.sharp,
                     (int(px - rx * ln), int(py - ry * ln)),
                     (int(px + rx * ln), int(py + ry * ln)),
                     _scale(chakra_color(0.97), 0.55 * e), 1, lineType=cv2.LINE_AA)

    # The disc the blades sweep through - a faint boundary only. Discrete radial
    # ticks around it read as a clock face, so the perimeter detail lives on the
    # blades instead.
    cv2.circle(layers.sharp, (cx, cy), max(2, int(blade_len * 0.99)),
               _scale(halo, 0.14 * e), 1, lineType=cv2.LINE_AA)

    # The original sphere stays visibly itself at the exact centre.
    draw_rasengan(layers, cx, cy, radius, frame_idx, charge_frac=1.0)


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
            draw_rasenshuriken(layers, self.x, self.y, self.radius, self.frame,
                               velocity=self.velocity, speed=speed_factor)
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
