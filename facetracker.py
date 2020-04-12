import cv2 as cv
import numpy as np

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def getOpticalFlowPtCoords(pt):
    if isinstance(pt, tuple):
        return int(pt[0][0][0][0]), int(pt[0][0][0][1])
    else:
        return int(pt[0][0][0]), int(pt[0][0][1])

class Point:
    __slots__ = ["x", "y"]
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def toTuple(self):
        return (self.x,self.y)
    def toCalcOpticalFlowPyrLKFeature(self):
        return (np.array([[(self.x, self.y)]], np.float32), )

class Rectangle:
    __slots__ = ["pt1", "pt2"]
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

class Face:
    __slots__ = ["rect", "pt"]
    def __init__(self, rect, pt):
        self.rect = rect
        self.pt = pt

class FaceTracker:
    def __init__(self, cap, faceClassifier):
        self.cap = cap
        self.faceClassifier = faceClassifier
        self.imgGray = None
        self.ptOld = None
        self.pt = None

    def initTracker(self):
        img, detections, self.imgGray = self.detectFaces()
        if len(detections) > 0:
            faces = self.getFaces(detections)
            self.pt = faces[0].pt.toCalcOpticalFlowPyrLKFeature()
            return True, img, self.pt
        else:
            return False, img, None
   
    def getFaces(self, detections):
        faces = []
        for (x,y,w,h) in detections:
            rect = Rectangle(Point(x, y), Point(x + w, y + h))
            pt = Point(x + int(w / 2), y + int(h / 3))
            faces.append(Face(rect, pt))
            break # just get first one
        return faces

    def labelFaces(self, img, faces):
        for face in faces:
            cv.rectangle(img, face.rect.pt1.toTuple(), face.rect.pt2.toTuple(), (0, 0, 255), 4)
            cv.circle(img, face.pt.toTuple(), 4, (0, 255, 255), -1)
            break # just get first one

    def detectFaces(self):
        ret, img = self.cap.read()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detections = self.faceClassifier.detectMultiScale(imgGray, 1.3, 5)
        return img, detections, imgGray
            
    def trackFaces(self):
        imgGrayOld = self.imgGray.copy()
        self.ptOld = self.pt[0]
        img, detections, self.imgGray = self.detectFaces()
        faces = self.getFaces(detections)
        self.labelFaces(img, faces)
        self.pt = cv.calcOpticalFlowPyrLK(imgGrayOld, self.imgGray, self.ptOld, None, **lk_params)
        return img, self.pt

    def getAnswer(self, xMovementOld, yMovementOld, movementThreshold):
        a,b = getOpticalFlowPtCoords(self.ptOld), getOpticalFlowPtCoords(self.pt)

        xMovement = xMovementOld + abs(a[0]-b[0])
        yMovement = yMovementOld + abs(a[1]-b[1])
        
        isAnswer = False
        answer = None

        if xMovement > movementThreshold:
            isAnswer = True
            answer = 'NO'

        elif yMovement > movementThreshold:
            isAnswer = True
            answer = 'YES'

        return isAnswer, answer, xMovement, yMovement
