"""
Rasengan & Rasenshuriken: weave the hand seals, then throw the jutsu.

Perform the six-seal sequence in order - Serpent, Tiger, Horse, Hare, Boar, Ram -
in front of the camera. The HUD lights up each seal as it registers. Complete the
sequence and a Rasengan spins into life in your palm and follows your hand.

Keep holding it and chakra builds. Once the charge bar fills, squeeze your hand
shut and the blue Rasengan tears open into a white four-bladed Rasenshuriken -
it stays one from then on, so you can open your hand again to throw it. Swing
outward fast to throw whichever you are holding; the Rasenshuriken detonates
into a dome of wind blades when it lands.

Seal recognition matches against templates recorded from your own hands, so it
needs training first:

    python main.py --train     record the six seals (writes seals.json)
    python main.py             cast

Press 'r' to drop the jutsu and reset the sequence. Press 'q' or ESC to quit.
"""
import math
import sys

import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands

import gestures as G
import effects as FX
import seals as S
import train as T

# --- tunables -------------------------------------------------------------
MAX_HANDS_TRACKED = 2
MATCH_DIST = 0.18          # normalized distance to keep matching a hand to its tracked slot
SLOT_TIMEOUT_FRAMES = 20   # frames a slot can go unmatched before its tracking resets
ANCHOR_EMA = 0.45          # how tightly the jutsu follows the palm it is riding on

MAX_CHARGE = 100.0
CHARGE_PER_FRAME = MAX_CHARGE / 75.0  # ~2.5s at 30fps to reach Rasenshuriken
CHARGE_DECAY = 2.0         # charge bled off per frame while no hand is visible
DROP_AFTER_LOST_FRAMES = 45  # jutsu fizzles out if your hands stay gone this long

ARM_FRAMES = 10            # grace period so the last seal's motion can't insta-throw
THROW_SPEED_MIN = 0.055    # normalized frame-to-frame palm speed that counts as a throw

SQUEEZE_RATIO = 1.55       # grip_ratio below this counts as squeezing (a light curl)
SQUEEZE_HOLD_FRAMES = 4    # debounce, so one noisy frame can't convert the jutsu
EMERGENCE_PER_FRAME = 1 / 14.0  # ~0.5s for the wind chakra to expand into a shuriken

BASE_RADIUS = 22.0
RADIUS_PER_CHARGE = 0.10

IDLE, RASENGAN, RASENSHURIKEN = "idle", "rasengan", "rasenshuriken"
TEXT = cv2.FONT_HERSHEY_SIMPLEX


def fresh_slot():
    """Per-hand identity tracking (used to match detections frame-to-frame)."""
    return {
        "last_pos": None,
        "missing_frames": 0,
    }


def fresh_jutsu():
    """The single jutsu being held, if any."""
    return {
        "state": IDLE,
        "pos": None,
        "charge": 0.0,
        "slot": None,      # which tracked hand it is riding on
        "age": 0,
        "lost_frames": 0,
        "squeeze_frames": 0,
        "emergence": 0.0,   # 0->1 as the wind chakra expands out of the sphere
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


def jutsu_radius(charge):
    return BASE_RADIUS + charge * RADIUS_PER_CHARGE


def draw_hud(frame, jutsu, tracker, live_match, trained, grip=None, frame_idx=0):
    """Seal sequence across the bottom, jutsu state and live match up top."""
    h, w = frame.shape[:2]

    if not trained:
        cv2.putText(frame, "no seal templates - run:  python main.py --train",
                    (16, 34), TEXT, 0.62, (80, 160, 255), 2, cv2.LINE_AA)
    elif jutsu["state"] == RASENSHURIKEN:
        cv2.putText(frame, "RASENSHURIKEN", (16, 34), TEXT, 0.78, (255, 250, 245), 2, cv2.LINE_AA)
    elif jutsu["state"] == RASENGAN:
        charged = jutsu["charge"] >= MAX_CHARGE
        pct = int(jutsu["charge"] / MAX_CHARGE * 100)
        cv2.putText(frame, f"RASENGAN  {pct}%", (16, 34), TEXT, 0.78, (255, 150, 60), 2, cv2.LINE_AA)
        bar_w = int(220 * jutsu["charge"] / MAX_CHARGE)
        cv2.rectangle(frame, (16, 44), (236, 54), (70, 70, 70), 1)
        if bar_w > 1:
            cv2.rectangle(frame, (17, 45), (16 + bar_w, 53), (255, 150, 60), -1)
        if charged:
            # Squeezing before this point does nothing on purpose, so say clearly
            # when it starts working - otherwise it just reads as broken.
            flash = 255 if (frame_idx // 8) % 2 == 0 else 140
            cv2.putText(frame, "SQUEEZE YOUR HAND", (16, 80), TEXT, 0.72,
                        (flash, flash, flash), 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "WEAVE THE SEALS", (16, 34), TEXT, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

    # Live grip reading - this is the number to quote if the squeeze threshold
    # needs adjusting for your hand.
    if grip is not None:
        squeezing = grip < SQUEEZE_RATIO
        cv2.putText(frame, f"grip {grip:0.2f} / {SQUEEZE_RATIO:0.2f}", (16, h - 62), TEXT, 0.45,
                    (120, 255, 160) if squeezing else (140, 140, 140), 1, cv2.LINE_AA)

    if live_match:
        cv2.putText(frame, live_match.upper(), (w - 170, 34), TEXT, 0.7, (120, 255, 160), 2, cv2.LINE_AA)

    # The sequence, lit up as far as you've got.
    if jutsu["state"] == IDLE:
        x = 16
        for i, name in enumerate(S.SEQUENCE):
            done = i < tracker.progress
            nxt = i == tracker.progress
            color = (120, 255, 160) if done else ((255, 200, 90) if nxt else (95, 95, 95))
            label = name.upper()
            cv2.putText(frame, label, (x, h - 40), TEXT, 0.52, color, 2 if (done or nxt) else 1, cv2.LINE_AA)
            x += cv2.getTextSize(label, TEXT, 0.52, 2)[0][0] + 10
            if i < len(S.SEQUENCE) - 1:
                cv2.putText(frame, ">", (x - 4, h - 40), TEXT, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
                x += 14

    cv2.putText(frame, "weave the seals  -  hold to charge  -  squeeze for Rasenshuriken  -  swing to throw   r=reset  q=quit",
                (16, h - 14), TEXT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


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

    if "--train" in sys.argv:
        T.run_training(cap, hands)
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        return

    recognizer = S.SealRecognizer.load()
    if recognizer is None:
        print("No seal templates found. Run 'python main.py --train' to record them first.")
    else:
        missing = recognizer.missing_seals()
        if missing:
            print("Warning: no templates recorded for: " + ", ".join(missing))

    tracker = S.SequenceTracker()
    slots = [fresh_slot() for _ in range(MAX_HANDS_TRACKED)]
    jutsu = fresh_jutsu()
    aura = FX.AuraParticles()
    projectiles = []

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

        layers = FX.GlowLayers(frame.shape)

        raw_hands = result.multi_hand_landmarks if result.multi_hand_landmarks else []
        detections = [(*G.palm_center(hlm.landmark), hlm.landmark) for hlm in raw_hands]

        for hlm in raw_hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hlm.landmark]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 2, (150, 150, 150), -1, cv2.LINE_AA)

        assigned = match_hands_to_slots(slots, detections)

        active = {}  # slot_idx -> (cx, cy, prev_pos, lm) for every hand seen this frame
        for i, slot in enumerate(slots):
            hit = assigned.get(i)
            if hit is None:
                slot["missing_frames"] += 1
                if slot["missing_frames"] > SLOT_TIMEOUT_FRAMES:
                    slots[i] = fresh_slot()
                continue

            cx, cy, lm = hit
            active[i] = (cx, cy, slot["last_pos"], lm)
            slot["missing_frames"] = 0
            slot["last_pos"] = (cx, cy)

        # --- seal weaving (only while empty-handed) ---------------------
        live_match = None
        grip = None
        if jutsu["state"] == IDLE and recognizer is not None:
            vector, n_hands = S.feature_from_result(result)
            stable, live_match, _dist = recognizer.update(vector, n_hands)
            if tracker.feed(stable):
                anchor = next(iter(active), None)
                pos = active[anchor][:2] if anchor is not None else (0.5, 0.5)
                jutsu.update({"state": RASENGAN, "pos": pos, "charge": 0.0,
                              "slot": anchor, "age": 0, "lost_frames": 0,
                              "squeeze_frames": 0, "emergence": 0.0})
                recognizer.reset()

        # --- holding a jutsu -------------------------------------------
        elif jutsu["state"] != IDLE:
            jutsu["age"] += 1

            # Stay on the hand it was formed in; fall back to any live hand.
            hand = active.get(jutsu["slot"])
            if hand is None and active:
                jutsu["slot"] = next(iter(active))
                hand = active[jutsu["slot"]]

            if hand is None:
                jutsu["lost_frames"] += 1
                jutsu["charge"] = max(0.0, jutsu["charge"] - CHARGE_DECAY)
                if jutsu["lost_frames"] > DROP_AFTER_LOST_FRAMES:
                    jutsu = fresh_jutsu()
                    tracker.reset()
            else:
                cx, cy, prev_pos, lm = hand
                jutsu["lost_frames"] = 0
                grip = G.grip_ratio(lm)

                px, py = jutsu["pos"]
                jutsu["pos"] = (px + (cx - px) * ANCHOR_EMA, py + (cy - py) * ANCHOR_EMA)

                speed = math.hypot(cx - prev_pos[0], cy - prev_pos[1]) if prev_pos else 0.0
                if jutsu["age"] > ARM_FRAMES and speed > THROW_SPEED_MIN and prev_pos:
                    vx, vy = cx - prev_pos[0], cy - prev_pos[1]
                    n = math.hypot(vx, vy) + 1e-6
                    ox, oy = jutsu["pos"]
                    projectiles.append(FX.JutsuBlast(
                        int(ox * w), int(oy * h), vx / n, vy / n, jutsu["state"],
                        speed=28, radius=int(jutsu_radius(jutsu["charge"]))))
                    jutsu = fresh_jutsu()
                    tracker.reset()
                else:
                    if jutsu["state"] == RASENGAN:
                        jutsu["charge"] = min(MAX_CHARGE, jutsu["charge"] + CHARGE_PER_FRAME)
                        # Squeeze converts it, but only once it's fully charged.
                        # It latches: opening your hand afterwards keeps the
                        # Rasenshuriken so you can open up to throw it.
                        if jutsu["charge"] >= MAX_CHARGE and grip is not None and grip < SQUEEZE_RATIO:
                            jutsu["squeeze_frames"] += 1
                            if jutsu["squeeze_frames"] >= SQUEEZE_HOLD_FRAMES:
                                jutsu["state"] = RASENSHURIKEN
                        else:
                            jutsu["squeeze_frames"] = 0
                    if jutsu["state"] == RASENSHURIKEN:
                        jutsu["emergence"] = min(1.0, jutsu["emergence"] + EMERGENCE_PER_FRAME)
                    color = FX.RASENSHURIKEN_COLOR if jutsu["state"] == RASENSHURIKEN else FX.RASENGAN_COLOR
                    aura.emit(int(cx * w), int(cy * h), 14, n=2, color=color)

        if jutsu["state"] != IDLE and jutsu["pos"] is not None:
            ox, oy = int(jutsu["pos"][0] * w), int(jutsu["pos"][1] * h)
            radius = jutsu_radius(jutsu["charge"])
            if jutsu["state"] == RASENSHURIKEN:
                FX.draw_rasenshuriken(layers, ox, oy, radius, frame_idx,
                                      emergence=jutsu["emergence"])
            else:
                FX.draw_rasengan(layers, ox, oy, radius, frame_idx,
                                 charge_frac=jutsu["charge"] / MAX_CHARGE)

        aura.update_and_draw(layers)

        next_projectiles = []
        for p in projectiles:
            p.update()
            if p.offscreen(w, h):
                if getattr(p, "detonates", False) and p.life <= 0:
                    next_projectiles.append(FX.WindDome(p.x, p.y))
                continue
            p.draw(layers)
            next_projectiles.append(p)
        projectiles = next_projectiles

        frame = layers.composite(frame)

        draw_hud(frame, jutsu, tracker, live_match, recognizer is not None, grip, frame_idx)

        cv2.imshow("Rasengan", frame)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            slots = [fresh_slot() for _ in range(MAX_HANDS_TRACKED)]
            jutsu = fresh_jutsu()
            projectiles = []
            tracker.reset()
            if recognizer is not None:
                recognizer.reset()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
