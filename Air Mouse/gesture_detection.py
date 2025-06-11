import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def _init_(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList, self.lmList = [], [], []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []  # Prevent errors if no hands are detected

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  # Check thumb position
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  # Check if finger is up
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera not detected. Try changing the index (0,1,2).")
        return  # Exit if the camera is not detected

    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if lmList:
            fingers = detector.fingersUp()
            print(f"Fingers Up: {fingers}")  # Prints the detected finger gestures

        cv2.imshow("Hand Tracking", img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()