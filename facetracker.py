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
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
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
    def __init__(self, cap, frameDimensions, faceClassifier, showLabels=False):
        self.cap = cap
        self.frameDimensions = frameDimensions
        self.faceClassifier = faceClassifier
        self.showLabels = showLabels
        self.isDebugInit = False
        self.imgGray = None
        self.ptOld = None
        self.pt = None
        self.lastKnownFace = None

    def initTracker(self):
        img, detections, self.imgGray = self.detectFaces()
        if len(detections) > 0:
            faces = self.getFaces(detections)
            self.pt = faces[0].pt.toCalcOpticalFlowPyrLKFeature()
            return True, img
        else:
            return False, img
   
    def getFaces(self, detections):
        faces = []
        for (x,y,w,h) in detections:
            rect = Rectangle(Point(x, y), Point(x + w, y + h))
            pt = Point(x + int(w / 2), y + int(h / 3))
            face = Face(rect, pt)
            faces.append(face)
            self.lastKnownFace = face
            break # just get first one
        return faces

    def labelFaces(self, img, faces):
        if self.isDebugInit and self.lastKnownFace != None:
            cv.line(img, (self.lastKnownFace.pt.x,0), (self.lastKnownFace.pt.x, self.frameDimensions[1]), (255, 255, 255), thickness=1)
            cv.line(img, (0, self.lastKnownFace.pt.y), (self.frameDimensions[0], self.lastKnownFace.pt.y), (255, 255, 255), thickness=1)
            cv.circle(img, self.lastKnownFace.pt.toTuple(), 4, (0, 0, 255), -1)
        else:
            for face in faces:
                cv.rectangle(img, face.rect.pt1.toTuple(), face.rect.pt2.toTuple(), (0, 0, 255), 4)
                cv.circle(img, face.pt.toTuple(), 4, (0, 255, 255), -1)
                break # just get first one

    def detectFaces(self):
        _, img = self.cap.read()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detections = self.faceClassifier.detectMultiScale(imgGray, 1.3, 5)
        return img, detections, imgGray
            
    def trackFaces(self):
        imgGrayOld = self.imgGray.copy()
        self.ptOld = self.pt[0]
        img, detections, self.imgGray = self.detectFaces()
        faces = self.getFaces(detections)
        if self.showLabels:
            self.labelFaces(img, faces)
        self.pt = cv.calcOpticalFlowPyrLK(imgGrayOld, self.imgGray, self.ptOld, None, **lk_params)
        a,b = getOpticalFlowPtCoords(self.ptOld), getOpticalFlowPtCoords(self.pt)
        xDelta = abs(a[0]-b[0])
        yDelta = abs(a[1]-b[1])
        return img, xDelta, yDelta

    def debug(self):
        while True:
            if not self.isDebugInit:
                self.isDebugInit, img = self.initTracker()
            else:
                img, detections, self.imgGray = self.detectFaces()
                faces = self.getFaces(detections)
                self.labelFaces(img, faces)

            cv.imshow('Face Tracker Testing', img)

            ch = cv.waitKey(1)
            if ch == 27:
                break
