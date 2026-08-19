"""Hand seal recognition by nearest-neighbour matching against recorded templates.

Seals like Serpent, Tiger and Ram press the palms together with interlocked
fingers, which is exactly where MediaPipe's landmarks get unreliable - it often
drops to a single detected hand or reports anatomically wrong joints. Rather
than hand-coding finger anatomy that those poses violate, we record what
MediaPipe actually outputs for the user's own hands and match against that.
Whatever it reports for a given seal is at least *consistent*, and consistency
is all a nearest-neighbour matcher needs.
"""
import json
import os

import numpy as np

SEAL_NAMES = ["serpent", "tiger", "horse", "hare", "boar", "ram"]
SEQUENCE = list(SEAL_NAMES)  # cast order happens to be the same list, in order

TEMPLATE_VERSION = 1
DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seals.json")

# --- tunables -------------------------------------------------------------
MATCH_MAX_DIST = 0.40      # mean per-landmark distance (in RMS-normalized units)
MATCH_MARGIN = 0.04        # how much closer the winner must be than the runner-up
SEAL_HOLD_FRAMES = 6       # frames a seal must stay the top match before it counts
SEQ_TIMEOUT_FRAMES = 120   # ~4s at 30fps of no progress resets the sequence


def pose_feature(hand_landmarks, handedness):
    """Turn this frame's hands into a (vector, n_hands) pose descriptor.

    hand_landmarks: list of MediaPipe landmark lists (one per detected hand).
    handedness: list of "Left"/"Right" labels, parallel to hand_landmarks.

    The descriptor is translation- and scale-invariant so it doesn't matter
    where you stand or how far from the camera you are. It is deliberately NOT
    rotation-invariant: orientation is what separates Boar from Ram.
    """
    if not hand_landmarks:
        return None, 0

    # Deterministic hand ordering so training and inference agree. The frame is
    # flipped before detection, so MediaPipe's labels read correctly for the user.
    order = sorted(
        range(len(hand_landmarks)),
        key=lambda i: (handedness[i] if i < len(handedness) else "", hand_landmarks[i][0].x),
    )

    pts = []
    for i in order:
        for p in hand_landmarks[i]:
            pts.append((p.x, p.y))  # z is per-hand-relative and too noisy to help

    cloud = np.asarray(pts, dtype=np.float32)
    cloud -= cloud.mean(axis=0)
    rms = float(np.sqrt((cloud ** 2).sum(axis=1).mean()))
    if rms < 1e-6:
        return None, 0
    cloud /= rms
    return cloud.reshape(-1), len(hand_landmarks)


def feature_from_result(result):
    """Convenience wrapper: MediaPipe Hands result -> (vector, n_hands)."""
    landmarks = result.multi_hand_landmarks or []
    handed = [h.classification[0].label for h in (result.multi_handedness or [])]
    return pose_feature([hlm.landmark for hlm in landmarks], handed)


def _mean_landmark_distance(templates, vector):
    """Mean per-landmark euclidean distance from `vector` to each row of `templates`."""
    diff = templates - vector[None, :]
    diff = diff.reshape(len(templates), -1, 2)
    return np.linalg.norm(diff, axis=2).mean(axis=1)


class SealRecognizer:
    """Matches pose descriptors against recorded seal templates.

    Templates are bucketed by how many hands MediaPipe saw. That bucketing is
    what makes this robust: if the model only finds one hand for half your Tiger
    frames, both the 1-hand and 2-hand variants get recorded, and a live 1-hand
    frame is only ever compared against 1-hand templates.
    """

    def __init__(self):
        self.samples = {name: [] for name in SEAL_NAMES}  # name -> list of (n_hands, vector)
        self._buckets = {}  # n_hands -> (stacked array, list of names)
        self._candidate = None
        self._streak = 0

    # --- template storage -------------------------------------------------
    def add_sample(self, name, vector, n_hands):
        self.samples[name].append((n_hands, np.asarray(vector, dtype=np.float32)))
        self._buckets = {}

    def clear(self, name):
        self.samples[name] = []
        self._buckets = {}

    def counts(self):
        return {name: len(v) for name, v in self.samples.items()}

    def is_trained(self):
        return any(self.samples[name] for name in SEAL_NAMES)

    def missing_seals(self):
        return [name for name in SEAL_NAMES if not self.samples[name]]

    def save(self, path=DEFAULT_TEMPLATE_PATH):
        data = {
            "version": TEMPLATE_VERSION,
            "seals": {
                name: [{"hands": n, "v": [round(float(x), 5) for x in vec]} for n, vec in samples]
                for name, samples in self.samples.items()
            },
        }
        with open(path, "w") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path=DEFAULT_TEMPLATE_PATH):
        """Load templates, or return None if there's no usable file."""
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            data = json.load(fh)
        if data.get("version") != TEMPLATE_VERSION:
            return None
        rec = cls()
        for name, samples in data.get("seals", {}).items():
            if name not in rec.samples:
                continue
            for s in samples:
                rec.samples[name].append((s["hands"], np.asarray(s["v"], dtype=np.float32)))
        return rec if rec.is_trained() else None

    # --- matching ---------------------------------------------------------
    def _bucket(self, n_hands):
        """Templates for a given hand count, stacked into one array for a single matmul-ish pass."""
        if n_hands not in self._buckets:
            vecs, names = [], []
            for name, samples in self.samples.items():
                for n, vec in samples:
                    if n == n_hands:
                        vecs.append(vec)
                        names.append(name)
            self._buckets[n_hands] = (np.stack(vecs) if vecs else None, names)
        return self._buckets[n_hands]

    def classify(self, vector, n_hands):
        """Return (name, distance, runner_up_distance), any of which may be None."""
        if vector is None or n_hands == 0:
            return None, None, None
        templates, names = self._bucket(n_hands)
        if templates is None or templates.shape[1] != vector.shape[0]:
            return None, None, None

        dists = _mean_landmark_distance(templates, vector)
        best = int(np.argmin(dists))
        best_name, best_dist = names[best], float(dists[best])

        other = [float(d) for d, nm in zip(dists, names) if nm != best_name]
        runner_up = min(other) if other else None
        return best_name, best_dist, runner_up

    def accepted(self, vector, n_hands):
        """The seal matched this frame, or None if it's too far off or ambiguous."""
        name, dist, runner_up = self.classify(vector, n_hands)
        if name is None or dist > MATCH_MAX_DIST:
            return None, dist
        # Reject ambiguity rather than coin-flipping between lookalikes (Tiger vs Ram).
        if runner_up is not None and runner_up - dist < MATCH_MARGIN:
            return None, dist
        return name, dist

    def update(self, vector, n_hands):
        """Feed a frame. Returns (stable_seal_or_None, live_match_or_None, distance)."""
        name, dist = self.accepted(vector, n_hands)
        if name is None or name != self._candidate:
            self._candidate = name
            self._streak = 1 if name else 0
        else:
            self._streak += 1
        stable = name if (name and self._streak >= SEAL_HOLD_FRAMES) else None
        return stable, name, dist

    def reset(self):
        self._candidate = None
        self._streak = 0


class SequenceTracker:
    """Tracks progress through SEQUENCE as stable seals come in."""

    def __init__(self, sequence=SEQUENCE, timeout_frames=SEQ_TIMEOUT_FRAMES):
        self.sequence = list(sequence)
        self.timeout_frames = timeout_frames
        self.progress = 0
        self.last_confirmed = None
        self.idle_frames = 0

    def reset(self):
        self.progress = 0
        self.last_confirmed = None
        self.idle_frames = 0

    def expected(self):
        return self.sequence[self.progress] if self.progress < len(self.sequence) else None

    def feed(self, stable_seal):
        """Advance on the expected seal. Returns True on the frame the sequence completes."""
        if stable_seal is None:
            self.last_confirmed = None
        elif stable_seal != self.last_confirmed:
            # A seal you're still holding shouldn't advance twice.
            self.last_confirmed = stable_seal
            if stable_seal == self.expected():
                self.progress += 1
                self.idle_frames = 0
                if self.progress >= len(self.sequence):
                    self.reset()
                    return True
            # Anything else is ignored rather than punished - the pass-through
            # poses your hands make between seals shouldn't reset you.

        if self.progress > 0:
            self.idle_frames += 1
            if self.idle_frames > self.timeout_frames:
                self.reset()
        return False
