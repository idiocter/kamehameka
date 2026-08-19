"""Recording mode: capture landmark templates for each hand seal.

Run with `python main.py --train`. Hold a seal, press its number, and a short
countdown gives you time to settle before a burst of frames is captured. Burst
capture rather than single-shot so natural hand jitter ends up in the templates.
"""
import cv2
import mediapipe as mp

import seals as S

COUNTDOWN_FRAMES = 45   # ~1.5s to settle into the seal before capture starts
BURST_FRAMES = 30       # samples captured per press
TEXT = cv2.FONT_HERSHEY_SIMPLEX

mp_hands = mp.solutions.hands


def run_training(cap, hands, path=S.DEFAULT_TEMPLATE_PATH):
    """Interactive template recorder. Returns True if templates were saved."""
    recognizer = S.SealRecognizer.load(path) or S.SealRecognizer()
    connections = mp_hands.HAND_CONNECTIONS

    target = None          # seal currently being recorded
    countdown = 0
    remaining = 0
    dirty = False
    status = "hold a seal, press its number to record"
    awaiting_clear = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        for hlm in (result.multi_hand_landmarks or []):
            pts = [(int(p.x * w), int(p.y * h)) for p in hlm.landmark]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (90, 90, 90), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 2, (170, 170, 170), -1, cv2.LINE_AA)

        vector, n_hands = S.feature_from_result(result)

        if target is not None:
            if countdown > 0:
                countdown -= 1
                secs = countdown / 30.0 + 0.05
                cv2.putText(frame, f"{target.upper()} in {secs:0.1f}", (w // 2 - 130, h // 2),
                            TEXT, 1.2, (60, 200, 255), 3, cv2.LINE_AA)
            elif remaining > 0:
                if vector is not None:
                    # Frames with no hand detected are skipped, not counted.
                    recognizer.add_sample(target, vector, n_hands)
                    remaining -= 1
                    dirty = True
                cv2.putText(frame, f"RECORDING {target.upper()}  {BURST_FRAMES - remaining}/{BURST_FRAMES}",
                            (w // 2 - 190, h // 2), TEXT, 0.9, (80, 255, 140), 2, cv2.LINE_AA)
                cv2.circle(frame, (w - 40, 40), 12, (60, 60, 255), -1, cv2.LINE_AA)
                if remaining == 0:
                    status = f"recorded {target}  -  press 's' to save"
                    target = None
            else:
                target = None

        counts = recognizer.counts()
        cv2.putText(frame, "SEAL TRAINING", (16, 30), TEXT, 0.7, (60, 200, 255), 2, cv2.LINE_AA)
        for i, name in enumerate(S.SEAL_NAMES):
            n = counts[name]
            color = (120, 255, 160) if n else (110, 110, 110)
            cv2.putText(frame, f"{i + 1}  {name:<9} {n:>3}", (16, 60 + i * 22),
                        TEXT, 0.55, color, 1, cv2.LINE_AA)

        hint = "x then 1-6 = clear a seal" if not awaiting_clear else "CLEAR WHICH? press 1-6 (esc cancels)"
        cv2.putText(frame, status, (16, h - 38), TEXT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"1-6 = record   s = save   {hint}   q = quit",
                    (16, h - 16), TEXT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Seal Training", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == 27:
            awaiting_clear = False
        elif key == ord('x'):
            awaiting_clear = True
            status = "clear which seal?"
        elif ord('1') <= key <= ord('6'):
            name = S.SEAL_NAMES[key - ord('1')]
            if awaiting_clear:
                recognizer.clear(name)
                awaiting_clear = False
                dirty = True
                status = f"cleared {name}"
            elif target is None:
                target = name
                countdown = COUNTDOWN_FRAMES
                remaining = BURST_FRAMES
                status = f"get into {name}..."
        elif key == ord('s'):
            recognizer.save(path)
            dirty = False
            missing = recognizer.missing_seals()
            status = ("saved  -  still missing: " + ", ".join(missing)) if missing else "saved all six seals"

    cv2.destroyWindow("Seal Training")
    if dirty:
        recognizer.save(path)
        print(f"Saved templates to {path}")
    return recognizer.is_trained()
