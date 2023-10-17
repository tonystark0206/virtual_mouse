import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
pinch_detected = False
palm_closed_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            index_tip = landmarks.landmark[8]
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            if prev_x is not None and prev_y is not None:
                delta_x = x - prev_x
                delta_y = y - prev_y
                pyautogui.move(delta_x, delta_y)

            prev_x, prev_y = x, y

            thumb_tip = landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            distance = cv2.norm((x, y), (thumb_x, thumb_y))

            if distance < 40:
                if not pinch_detected:
                    pyautogui.click()
                    pinch_detected = True
            else:
                pinch_detected = False

            # Check for a "palm closed" gesture
            if landmarks.landmark[0].y > landmarks.landmark[5].y:
                if not palm_closed_detected:
                    pyautogui.scroll(3)  # Scroll up
                    palm_closed_detected = True
            else:
                palm_closed_detected = False

        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
