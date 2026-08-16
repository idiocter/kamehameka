"""
Energy Orb: circle your hand to conjure a glowing ball of ki, then throw it.
"""
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

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
        for hlm in raw_hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hlm.landmark]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 2, (150, 150, 150), -1, cv2.LINE_AA)

        cv2.imshow("Energy Orb", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
