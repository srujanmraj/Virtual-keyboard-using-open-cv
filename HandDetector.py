import mediapipe as mp
import numpy as np
import cv2
import time


class HandTracker():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,self.detectionCon, self.trackCon)        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, handNo = 0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTracker()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)








if __name__ == "__main__":
    main()