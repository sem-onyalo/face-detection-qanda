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
        cv.circle(img, tracker.pt.toTuple(), 4, (0, 255, 255), -1)
        break # just get first one

def detectFaces(cap, faceClassifier):
    ret, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(imgGray, 1.3, 5)
    return img, faces, imgGray
    
# function to get coordinates
def getOpticalFlowPtCoords(pt):
    if isinstance(pt, tuple):
        return int(pt[0][0][0][0]), int(pt[0][0][0][1])
    else:
        return int(pt[0][0][0]), int(pt[0][0][1])

def addText(img, text, textColor=(238,238,238), textScale=1, textThickness=2, textPadding=[0,0,0,0]):
    topPad = textPadding[0]
    bottomPad = textPadding[2]
    res = "640,480" # TODO pull from argparse.ArgumentParser()
    font = cv.FONT_HERSHEY_SIMPLEX
    textSize = cv.getTextSize(text, font, textScale, textThickness)
    textWidth = textSize[0][0]
    textHeight = textSize[0][1]
    frameDim = list(map(int, res.split(',')))
    xPos = int(round((frameDim[0] / 1 - textWidth) / 2))
    yPos = 0
    if topPad > 0:
        yPos = topPad + textHeight
    elif bottomPad > 0:
        yPos = frameDim[1] - textHeight - bottomPad
    cv.putText(img, text, (xPos, yPos), font, textScale, textColor, textThickness, cv.LINE_AA)

def showQuestion(img, text):
    padding = [20,0,0,0]
    addText(img, text, (0,0,255), textPadding=padding)

def showAnswer(img, text):
    padding = [0,0,10,0]
    addText(img, text, (0,0,255), textPadding=padding)

def initApp(faces):
    if len(faces) > 0:
        trackers = getFaceTrackers(faces)
        pt = trackers[0].pt.toCalcOpticalFlowPyrLKFeature()
        return True, pt
    else:
        return False, None
        
def trackFaces(img, faces, imgGrayOld, imgGray, ptOld):
    trackers = getFaceTrackers(faces)
    labelFaces(img, trackers)
    pt = cv.calcOpticalFlowPyrLK(imgGrayOld, imgGray, ptOld, None, **lk_params)
    return pt

def getAnswer(ptOld, pt, xMovementOld, yMovementOld, movementThreshold):
    a,b = getOpticalFlowPtCoords(ptOld), getOpticalFlowPtCoords(pt)

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

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

testQuestions = [
    'Is this a cool app?',
    'Was it difficult for you to build?',
    'Does this app have the potential to be creepy?'
]

cap = cv.VideoCapture(0)
faceClassifier = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# constants
movementThreshold = 175
showQuestionMaxTime = 60
showAnswerMaxTime = 60

imgGray = None
pt = None

# App states
stateInit = 0
stateShowQuestion = 1
stateGetAnswer = 2
stateShowAnswer = 3
stateEnd = 4

# Run vars
questions = testQuestions
currentState = stateInit
showQuestionTime = 0
showAnswerTime = 0
questionIndex = 0
answer = False
xMovement = 0
yMovement = 0

# Run app
while True:
    if currentState == stateInit:
        img, faces, imgGray = detectFaces(cap, faceClassifier)
        isInit, pt = initApp(faces)
        if isInit:
            currentState = stateShowQuestion
            showQuestionTime = showQuestionMaxTime

    elif currentState == stateShowQuestion:
        ret, img = cap.read()
        showQuestion(img, questions[questionIndex])
        showQuestionTime -= 1
        if showQuestionTime == 0:
            currentState = stateGetAnswer
            showQuestionTime = showQuestionMaxTime

    elif currentState == stateGetAnswer:
        imgGrayOld = imgGray.copy()
        ptOld = pt[0]
        img, faces, imgGray = detectFaces(cap, faceClassifier)

        pt = trackFaces(img, faces, imgGrayOld, imgGray, ptOld)

        showQuestion(img, questions[questionIndex])

        isAnswer, answer, xMovement, yMovement = getAnswer(ptOld, pt, xMovement, yMovement, movementThreshold)
        
        cv.putText(img, 'xMovement: ' + str(xMovement), (50,100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv.putText(img, 'yMovement: ' + str(yMovement), (50,150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if isAnswer != False:
            currentState = stateShowAnswer
            showAnswerTime = showAnswerMaxTime

    elif currentState == stateShowAnswer:
        ret, img = cap.read()
        showQuestion(img, questions[questionIndex])
        showAnswer(img, answer)
       
    cv.imshow('image', img)

    ch = cv.waitKey(1)
    if ch == 27:
        break
    