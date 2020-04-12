import cv2 as cv

# Constants
movementThreshold = 175
movementDeltaThreshold = 10
showQuestionMaxTime = 60
showAnswerMaxTime = 60

debugMovementThreshold = 250
debugShowAnswerMaxTime = 90

# App states
stateDebug = -1
stateInit = 0
stateShowQuestion = 1
stateGetAnswer = 2
stateShowAnswer = 3
stateEnd = 4

class App:
    def __init__(self, cap, faceTracker, videoManager, initState, questions, frameDimensions):
        self.cap = cap
        self.questions = questions
        self.currentState = initState
        self.faceTracker = faceTracker
        self.videoManager = videoManager
        self.frameDimensions = frameDimensions
        
        self.currentQuestionIndex = 0
        self.showQuestionTime = 0
        self.showAnswerTime = 0
        self.xMovement = 0
        self.yMovement = 0
        self.answer = None

        self.debugShowAnswerTime = debugShowAnswerMaxTime
        self.isDebugInit = False

    def checkAnswer(self, xDelta, yDelta):
        if xDelta > movementDeltaThreshold or yDelta > movementDeltaThreshold:
            if xDelta > yDelta:
                self.xMovement = self.xMovement + xDelta
            elif yDelta > xDelta:
                self.yMovement = self.yMovement + yDelta

        if self.xMovement > movementThreshold:
            self.answer = 'NO'

        elif self.yMovement > movementThreshold:
            self.answer = 'YES'

    def run(self):
        while True:
            if self.currentState == stateInit:
                isInit, img = self.faceTracker.initTracker()
                if isInit:
                    self.currentState = stateShowQuestion
                    self.showQuestionTime = showQuestionMaxTime

            elif self.currentState == stateShowQuestion:
                _, img = self.cap.read()
                self.videoManager.showQuestion(img, self.questions[self.currentQuestionIndex], self.frameDimensions)
                self.showQuestionTime -= 1
                if self.showQuestionTime == 0:
                    self.currentState = stateGetAnswer
                    self.xMovement = 0
                    self.yMovement = 0
                    self.answer = None

            elif self.currentState == stateGetAnswer:
                img, xDelta, yDelta = self.faceTracker.trackFaces()
                self.videoManager.showQuestion(img, self.questions[self.currentQuestionIndex], self.frameDimensions)
                self.checkAnswer(xDelta, yDelta)
                
                if self.answer != None:
                    self.currentState = stateShowAnswer
                    self.showAnswerTime = showAnswerMaxTime

            elif self.currentState == stateShowAnswer:
                _, img = self.cap.read()
                self.videoManager.showQuestion(img, self.questions[self.currentQuestionIndex], self.frameDimensions)
                self.videoManager.showAnswer(img, self.answer, self.frameDimensions)
                self.showAnswerTime -= 1
                if self.showAnswerTime == 0:
                    self.currentQuestionIndex += 1
                    if self.currentQuestionIndex < len(self.questions):
                        self.currentState = stateShowQuestion
                        self.showQuestionTime = showQuestionMaxTime
                    else:
                        self.currentState = stateEnd

            elif self.currentState == stateEnd:
                _, img = self.cap.read()

            elif self.currentState == stateDebug:
                if not self.isDebugInit:
                    self.isDebugInit, img = self.faceTracker.initTracker()
                else:
                    img, xDelta, yDelta = self.faceTracker.trackFaces()
                    self.checkAnswer(xDelta, yDelta)

                    if self.answer != None and self.debugShowAnswerTime > 0:
                        self.videoManager.showAnswer(img, self.answer, self.frameDimensions)
                        self.debugShowAnswerTime -= 1

                    if self.debugShowAnswerTime == 0:
                        self.debugShowAnswerTime = debugShowAnswerMaxTime
                        self.xMovement = 0
                        self.yMovement = 0
                        self.answer = None

                    cv.putText(img, 'xMovement: ' + str(self.xMovement), (50, self.frameDimensions[1] - 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv.putText(img, 'yMovement: ' + str(self.yMovement), (50, self.frameDimensions[1] - 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    cv.putText(img, 'xDelta: ' + str(xDelta), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv.putText(img, 'yDelta: ' + str(yDelta), (50, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
            cv.imshow('Q&A with Face Detection', img)

            ch = cv.waitKey(1)
            if ch == 27:
                break
