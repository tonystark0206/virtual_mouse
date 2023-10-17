import cv2
import numpy as np
import pyautogui

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up parameters for the hand detection
hand_cascade = cv2.CascadeClassifier(r'D:\python-opencv-detect-master\haarcascade_hand.xml')  # You need to provide the correct path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for hand detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Calculate the center of the detected hand
        hand_center_x = x + w // 2
        hand_center_y = y + h // 2

        # Calculate the screen resolution (you might need to adjust these values)
        screen_width, screen_height = pyautogui.size()

        # Map the hand position to the screen resolution
        target_x = int(hand_center_x * screen_width / frame.shape[1])
        target_y = int(hand_center_y * screen_height / frame.shape[0])

        # Move the mouse pointer to the detected hand position
        pyautogui.moveTo(target_x, target_y)

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
