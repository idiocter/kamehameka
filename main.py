"""
Energy Orb: circle your hand to conjure a glowing ball of ki, then throw it.
"""
import math

import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands

import gestures as G

# --- tunables -------------------------------------------------------------
MAX_HANDS_TRACKED = 2
MATCH_DIST = 0.18          # normalized distance to keep matching a hand to its tracked slot
SLOT_TIMEOUT_FRAMES = 20   # frames a slot can go unmatched before its tracking resets
CENTROID_EMA = 0.03        # how slowly the orb's center follows the circling hand


def fresh_slot():
    return {
        "last_pos": None,
        "centroid": None,
        "missing_frames": 0,
    }


def match_hands_to_slots(slots, detections):
    """detections: list of (cx, cy, lm). Returns dict slot_idx -> (cx, cy, lm)."""
    unmatched = list(range(len(detections)))
    assigned = {}

    for i, slot in enumerate(slots):
        if slot["last_pos"] is None or not unmatched:
            continue
        lx, ly = slot["last_pos"]
        best_j, best_d = None, MATCH_DIST
        for j in unmatched:
            cx, cy, _ = detections[j]
            d = math.hypot(cx - lx, cy - ly)
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            assigned[i] = detections[best_j]
            unmatched.remove(best_j)

    for i, slot in enumerate(slots):
        if i in assigned or slot["last_pos"] is not None:
            continue
        if unmatched:
            assigned[i] = detections[unmatched.pop(0)]

    return assigned


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS_TRACKED,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    slots = [fresh_slot() for _ in range(MAX_HANDS_TRACKED)]
    connections = mp_hands.HAND_CONNECTIONS

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        raw_hands = result.multi_hand_landmarks if result.multi_hand_landmarks else []
        detections = [(*G.palm_center(hlm.landmark), hlm.landmark) for hlm in raw_hands]

        for hlm in raw_hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hlm.landmark]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 2, (150, 150, 150), -1, cv2.LINE_AA)

        assigned = match_hands_to_slots(slots, detections)

        for i, slot in enumerate(slots):
            hit = assigned.get(i)
            if hit is None:
                slot["missing_frames"] += 1
                if slot["missing_frames"] > SLOT_TIMEOUT_FRAMES:
                    slots[i] = fresh_slot()
                continue

            cx, cy, _lm = hit
            slot["missing_frames"] = 0

            if slot["centroid"] is None:
                slot["centroid"] = (cx, cy)
            cenx, ceny = slot["centroid"]
            cenx += (cx - cenx) * CENTROID_EMA
            ceny += (cy - ceny) * CENTROID_EMA
            slot["centroid"] = (cenx, ceny)

            slot["last_pos"] = (cx, cy)

        cv2.imshow("Energy Orb", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
