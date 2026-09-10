"""Hand landmark helpers."""
import math

# Landmark indices (MediaPipe Hands topology)
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# The thumb is left out - it curls least predictably and folds across the palm,
# which adds noise without helping tell an open hand from a closed one.
GRIP_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def palm_center(lm):
    xs = [lm[i].x for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    ys = [lm[i].y for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def palm_size_pixels(lm, width, height):
    """Wrist-to-middle-knuckle distance in the camera's pixel coordinates."""
    return math.hypot((lm[MIDDLE_MCP].x - lm[WRIST].x) * width,
                      (lm[MIDDLE_MCP].y - lm[WRIST].y) * height)


def throw_motion(current, previous, width, height, step=1.0):
    """Return pixel direction and screen-relative speed per 30 Hz tick."""
    if previous is None:
        return 0.0, 0.0, 0.0
    vx = (current[0] - previous[0]) * width
    vy = (current[1] - previous[1]) * height
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return 0.0, 0.0, 0.0
    return vx / length, vy / length, length / (max(width, height) * max(step, 0.01))


def grip_ratio(lm):
    """How open the hand is: mean fingertip distance from the palm centre,
    measured in units of palm size. Roughly 2.2 open, 1.0 in a fist.

    Palm size (wrist to middle MCP) barely changes as the fingers curl, so using
    it as the denominator cancels out how near the camera the hand is.
    """
    cx, cy = palm_center(lm)
    palm = math.hypot(lm[MIDDLE_MCP].x - lm[WRIST].x, lm[MIDDLE_MCP].y - lm[WRIST].y)
    if palm < 1e-6:
        return None
    reach = sum(math.hypot(lm[i].x - cx, lm[i].y - cy) for i in GRIP_TIPS) / len(GRIP_TIPS)
    return reach / palm
