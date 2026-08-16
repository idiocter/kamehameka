"""Hand landmark helpers."""

# Landmark indices (MediaPipe Hands topology)
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


def palm_center(lm):
    xs = [lm[i].x for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    ys = [lm[i].y for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    return sum(xs) / len(xs), sum(ys) / len(ys)
