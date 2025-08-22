import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None

# Default color
current_color = (255, 0, 0)  # Blue

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            # Index finger tip = landmark 8
            x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            # Draw line as finger moves
            if prev_x is None and prev_y is None:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)
            prev_x, prev_y = x, y

            # Draw circle on fingertip
            cv2.circle(frame, (x, y), 8, current_color, -1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None  # Reset if no hand detected

    # Merge canvas with webcam feed
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, "Press R/G/B/Y/P for colors |And C to clear | ESC to exit from tab"
    "Exited...*", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Air Writing with Colors", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC to exit
        break
    elif key & 0xFF == ord('c'):  # Clear
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    elif key & 0xFF == ord('r'):
        current_color = (0, 0, 255)  # Red
    elif key & 0xFF == ord('g'):
        current_color = (0, 255, 0)  # Green
    elif key & 0xFF == ord('b'):
        current_color = (255, 0, 0)  # Blue
    elif key & 0xFF == ord('y'):
        current_color = (0, 255, 255)  # Yellow
    elif key & 0xFF == ord('p'):
        current_color = (255, 0, 255)  # Purple

cap.release()
cv2.destroyAllWindows()