"""DBZ-style visual effects rendered with OpenCV, additive-blended onto frames."""
import numpy as np


def _glow_layer(shape):
    return np.zeros(shape, dtype=np.float32)


def blend_additive(frame, glow):
    """Additively blend a float32 BGR glow layer onto a uint8 frame."""
    out = frame.astype(np.float32) + glow
    return np.clip(out, 0, 255).astype(np.uint8)
