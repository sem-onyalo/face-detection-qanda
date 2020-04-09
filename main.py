import cv2 as cv
import numpy as np

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

class FaceTracker:
    __slots__ = ["rect", "pt"]
    def __init__(self, rect, pt):
        self.rect = rect
        self.pt = pt

def getFaceTrackers(faces):
    trackers = []
    for (x,y,w,h) in faces:
        rect = Rectangle(Point(x, y), Point(x + w, y + h))
        pt = Point(x + int(w / 2), y + int(h / 3))
        trackers.append(FaceTracker(rect, pt))
        break # just get first one
    return trackers

def labelFaces(img, trackers):
    for tracker in trackers:
        cv.rectangle(img, tracker.rect.pt1.toTuple(), tracker.rect.pt2.toTuple(), (0, 0, 255), 4)
        cv.circle(img, tracker.pt.toTuple(), 5, (0, 255, 255), -1)
        break # just get first one

def detectFaces(cap, faceClassifier):
    ret, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(imgGray, 1.3, 5)
    return img, faces, imgGray
    
# function to get coordinates
def get_coords(pt):
    if isinstance(pt, tuple):
        return int(pt[0][0][0][0]), int(pt[0][0][0][1])
    else:
        return int(pt[0][0][0]), int(pt[0][0][1])

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture(0)
faceClassifier = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# constants
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 75
# vars
gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60 # number of frames a gesture is shown
imgGray = None
pt = None
isFaceTrackerInit = False

# do it
while True:

    if isFaceTrackerInit == False:
        img, faces, imgGray = detectFaces(cap, faceClassifier)

        if len(faces) > 0:
            trackers = getFaceTrackers(faces)
            pt = trackers[0].pt.toCalcOpticalFlowPyrLKFeature()
            isFaceTrackerInit = True

    else:
        imgGrayOld = imgGray.copy()
        ptOld = pt[0]

        img, faces, imgGray = detectFaces(cap, faceClassifier)

        trackers = getFaceTrackers(faces)
        labelFaces(img, trackers)

        pt = cv.calcOpticalFlowPyrLK(imgGrayOld, imgGray, ptOld, None, **lk_params)
        
        a,b = get_coords(ptOld), get_coords(pt)
        x_movement += abs(a[0]-b[0])
        y_movement += abs(a[1]-b[1])
        
        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv.putText(img, text, (50,50 ), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv.putText(img, text, (50,100 ), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if x_movement > gesture_threshold:
            gesture = 'No'
        if y_movement > gesture_threshold:
            gesture = 'Yes'
        if gesture and gesture_show > 0:
            cv.putText(img, 'Gesture Detected: ' + gesture, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            gesture_show -= 1
        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 60 # number of frames a gesture is shown
       
    cv.imshow('image', img)

    ch = cv.waitKey(1)
    if ch == 27:
        break
    