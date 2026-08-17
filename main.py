"""
Energy Orb: circle your hand(s) to conjure a single glowing ball of ki, then throw it.

Move a hand in a circular motion in front of the camera - like winding up
for a Kamehameha - and a glowing orb forms at the center of the circle,
growing as you keep circling it. Use both hands together, circling in sync
around the space between them, to charge the very same orb faster - it's
always a single shared ball no matter how many hands are feeding it, never
more than one at a time. Once it's charged, swing a hand outward fast to
throw it; it flies off in the direction of the swing and fades away.

Press 'r' to clear the orb/charge. Press 'q' or ESC to quit.
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
CENTROID_EMA = 0.03        # how slowly the orb's center follows the circling hand(s)
MIN_LOOP_RADIUS = 0.045    # normalized distance from center required to count as "circling"
CHARGE_PER_REV = 18        # charge gained per full revolution, per hand
CHARGE_PER_RADIAN = CHARGE_PER_REV / (2 * math.pi)
MAX_CHARGE = 150
CHARGE_DECAY = 2.5         # charge lost per frame while nothing is circling
MIN_CHARGE_TO_THROW = 30
THROW_SPEED_MIN = 0.055    # normalized frame-to-frame speed that counts as a throwing swing
THROW_RADIUS_MIN = 0.09    # must swing out this far from the orb's center to release it

ORB_COLOR = (60, 200, 255)  # BGR - single shared orb, single color


def fresh_slot():
    """Per-hand identity tracking (used to match detections frame-to-frame)."""
    return {
        "last_pos": None,
        "prev_angle": None,
        "missing_frames": 0,
    }


def fresh_orb_state():
    """The single shared orb that all tracked hands feed into."""
    return {
        "centroid": None,
        "charge": 0.0,
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
    orb_state = fresh_orb_state()
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

        active = []  # (slot_idx, cx, cy, prev_pos) for every hand seen this frame
        for i, slot in enumerate(slots):
            hit = assigned.get(i)
            if hit is None:
                slot["missing_frames"] += 1
                slot["prev_angle"] = None
                if slot["missing_frames"] > SLOT_TIMEOUT_FRAMES:
                    slots[i] = fresh_slot()
                continue

            cx, cy, _lm = hit
            prev_pos = slot["last_pos"]
            slot["missing_frames"] = 0
            active.append((i, cx, cy, prev_pos))
            slot["last_pos"] = (cx, cy)

        if not active:
            orb_state["centroid"] = None
            orb_state["charge"] = max(0.0, orb_state["charge"] - CHARGE_DECAY)
        else:
            # the orb always sits at the midpoint of however many hands are feeding it
            target_x = sum(a[1] for a in active) / len(active)
            target_y = sum(a[2] for a in active) / len(active)
            if orb_state["centroid"] is None:
                orb_state["centroid"] = (target_x, target_y)
            else:
                cx0, cy0 = orb_state["centroid"]
                cx0 += (target_x - cx0) * CENTROID_EMA
                cy0 += (target_y - cy0) * CENTROID_EMA
                orb_state["centroid"] = (cx0, cy0)

            cenx, ceny = orb_state["centroid"]
            thrown = False
            any_circling = False

            for i, cx, cy, prev_pos in active:
                slot = slots[i]
                dx, dy = cx - cenx, cy - ceny
                radius = math.hypot(dx, dy)
                speed = math.hypot(cx - prev_pos[0], cy - prev_pos[1]) if prev_pos else 0.0

                if (not thrown and prev_pos and orb_state["charge"] >= MIN_CHARGE_TO_THROW
                        and radius > THROW_RADIUS_MIN and speed > THROW_SPEED_MIN):
                    vx, vy = cx - prev_pos[0], cy - prev_pos[1]
                    n = math.hypot(vx, vy) + 1e-6
                    px, py = int(cenx * w), int(ceny * h)
                    orb_radius = 12 + orb_state["charge"] * 0.55
                    orbs.append(FX.KiBlast(px, py, vx / n, vy / n, ORB_COLOR, speed=26, radius=int(orb_radius)))
                    orb_state["charge"] = 0.0
                    for s in slots:
                        s["prev_angle"] = None
                    thrown = True
                    continue

                if radius > MIN_LOOP_RADIUS:
                    any_circling = True
                    angle = math.atan2(dy, dx)
                    if slot["prev_angle"] is not None:
                        delta = angle - slot["prev_angle"]
                        delta = (delta + math.pi) % (2 * math.pi) - math.pi
                        orb_state["charge"] = min(MAX_CHARGE, orb_state["charge"] + abs(delta) * CHARGE_PER_RADIAN)
                    slot["prev_angle"] = angle
                    if not thrown:
                        aura.emit(int(cx * w), int(cy * h), 10, n=2, color=ORB_COLOR)
                else:
                    slot["prev_angle"] = None

            if not any_circling and not thrown:
                orb_state["charge"] = max(0.0, orb_state["charge"] - CHARGE_DECAY)

            if not thrown and orb_state["charge"] > 0:
                cenpx, cenpy = int(cenx * w), int(ceny * h)
                orb_radius = 12 + orb_state["charge"] * 0.55
                charge_frac = orb_state["charge"] / MAX_CHARGE
                FX.draw_charging_lightning(glow, cenpx, cenpy, orb_radius, charge_frac, ORB_COLOR, frame_idx)
                FX.draw_charge_orb(glow, cenpx, cenpy, orb_radius, ORB_COLOR, frame_idx)

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

        cv2.putText(frame, "circle one or both hands to charge the orb - swing outward to throw  r=reset  q=quit",
                    (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Energy Orb", frame)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            slots = [fresh_slot() for _ in range(MAX_HANDS_TRACKED)]
            orb_state = fresh_orb_state()
            orbs = []

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
