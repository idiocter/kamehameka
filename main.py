"""
Energy Orb: circle your hand to conjure a glowing ball of ki, then throw it.
"""
import math

import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands

import gestures as G
import effects as FX

# --- tunables -------------------------------------------------------------
MAX_HANDS_TRACKED = 2
MATCH_DIST = 0.18          # normalized distance to keep matching a hand to its tracked slot
SLOT_TIMEOUT_FRAMES = 20   # frames a slot can go unmatched before its tracking resets
CENTROID_EMA = 0.03        # how slowly the orb's center follows the circling hand
MIN_LOOP_RADIUS = 0.045    # normalized distance from center required to count as "circling"
CHARGE_PER_REV = 18        # charge gained per full revolution
MAX_CHARGE = 150
CHARGE_DECAY = 2.5         # charge lost per frame while not actively circling
MIN_CHARGE_TO_THROW = 15
THROW_SPEED_MIN = 0.03     # normalized frame-to-frame speed that counts as a throwing swing
THROW_RADIUS_MIN = 0.05    # must swing out this far from the orb's center to release it

ORB_COLORS = [(60, 200, 255), (255, 170, 60)]  # BGR, per tracked hand slot


def fresh_slot():
    return {
        "last_pos": None,
        "centroid": None,
        "prev_angle": None,
        "swept_angle": 0.0,
        "charge": 0.0,
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
    aura = FX.AuraParticles()
    orbs = []

    frame_idx = 0
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

        glow = FX._glow_layer(frame.shape)

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
            color = ORB_COLORS[i % len(ORB_COLORS)]
            hit = assigned.get(i)
            if hit is None:
                slot["missing_frames"] += 1
                slot["charge"] = max(0.0, slot["charge"] - CHARGE_DECAY)
                if slot["missing_frames"] > SLOT_TIMEOUT_FRAMES:
                    slots[i] = fresh_slot()
                continue

            cx, cy, _lm = hit
            prev_pos = slot["last_pos"]
            slot["missing_frames"] = 0

            if slot["centroid"] is None:
                slot["centroid"] = (cx, cy)
            cenx, ceny = slot["centroid"]
            cenx += (cx - cenx) * CENTROID_EMA
            ceny += (cy - ceny) * CENTROID_EMA
            slot["centroid"] = (cenx, ceny)

            dx, dy = cx - cenx, cy - ceny
            radius = math.hypot(dx, dy)

            if radius > MIN_LOOP_RADIUS:
                angle = math.atan2(dy, dx)
                if slot["prev_angle"] is not None:
                    delta = angle - slot["prev_angle"]
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    slot["swept_angle"] += abs(delta)
                slot["prev_angle"] = angle
            else:
                slot["prev_angle"] = None
                slot["charge"] = max(0.0, slot["charge"] - CHARGE_DECAY)

            while slot["swept_angle"] >= 2 * math.pi:
                slot["swept_angle"] -= 2 * math.pi
                slot["charge"] = min(MAX_CHARGE, slot["charge"] + CHARGE_PER_REV)

            cenpx, cenpy = int(cenx * w), int(ceny * h)
            px, py = int(cx * w), int(cy * h)
            orb_radius = 12 + slot["charge"] * 0.55
            speed = math.hypot(cx - prev_pos[0], cy - prev_pos[1]) if prev_pos else 0.0

            if (prev_pos and slot["charge"] >= MIN_CHARGE_TO_THROW
                    and radius > THROW_RADIUS_MIN and speed > THROW_SPEED_MIN):
                vx, vy = cx - prev_pos[0], cy - prev_pos[1]
                n = math.hypot(vx, vy) + 1e-6
                orbs.append(FX.KiBlast(px, py, vx / n, vy / n, color, speed=26, radius=int(orb_radius)))
                slot["charge"] = 0.0
                slot["swept_angle"] = 0.0
                slot["prev_angle"] = None
                slot["centroid"] = (cx, cy)
            elif slot["charge"] > 0:
                FX.draw_charge_orb(glow, cenpx, cenpy, orb_radius, color, frame_idx)
                if radius > MIN_LOOP_RADIUS:
                    aura.emit(px, py, 10, n=2, color=color)

            slot["last_pos"] = (cx, cy)

        aura.update_and_draw(glow)

        next_orbs = []
        for orb in orbs:
            orb.update()
            if orb.offscreen(w, h):
                continue
            orb.draw(glow)
            next_orbs.append(orb)
        orbs = next_orbs

        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=6, sigmaY=6)
        frame = FX.blend_additive(frame, glow)

        cv2.putText(frame, "circle a hand to charge an orb - swing it outward to throw  r=reset  q=quit",
                    (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Energy Orb", frame)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            slots = [fresh_slot() for _ in range(MAX_HANDS_TRACKED)]
            orbs = []

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
