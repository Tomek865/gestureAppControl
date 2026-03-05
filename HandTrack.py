import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cv2.flip(img, 0)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
