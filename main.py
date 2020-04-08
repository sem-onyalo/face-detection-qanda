import cv2 as cv

class Point:
    __slots__ = ["x", "y"]
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def toTuple(self):
        return (self.x,self.y)

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
    return trackers

def labelFaces(img, trackers):
    for tracker in trackers:
        cv.rectangle(img, tracker.rect.pt1.toTuple(), tracker.rect.pt2.toTuple(), (0, 0, 255), 4)
        cv.circle(img, tracker.pt.toTuple(), 6, (0, 255, 255), -1)

#capture source video
cap = cv.VideoCapture(0)

#path to face cascde
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# do it
while True:
    ret, img = cap.read()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    trackers = getFaceTrackers(faces)
    labelFaces(img, trackers)
    
    cv.imshow('image',img)

    ch = cv.waitKey(1)
    if ch == 27:
        break
