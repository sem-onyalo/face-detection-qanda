import cv2 as cv

# Constants
movementThreshold = 175
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
    def __init__(self, cap, tracker, videoManager, initState, questions, frameDimensions):
        self.cap = cap
        self.tracker = tracker
        self.questions = questions
        self.currentState = initState
        self.videoManager = videoManager
        self.frameDimensions = frameDimensions
        
        self.currentQuestionIndex = 0
        self.showQuestionTime = 0
        self.showAnswerTime = 0
        self.answer = False
        self.xMovement = 0
        self.yMovement = 0

        self.debugShowAnswerTime = debugShowAnswerMaxTime
        self.isDebugInit = False

    def run(self):
        while True:
            if self.currentState == stateInit:
                isInit, img, pt = self.tracker.initTracker()
                if isInit:
                    self.currentState = stateShowQuestion
                    self.showQuestionTime = showQuestionMaxTime

            elif self.currentState == stateShowQuestion:
                ret, img = self.cap.read()
                self.videoManager.showQuestion(img, self.questions[self.currentQuestionIndex], self.frameDimensions)
                self.showQuestionTime -= 1
                if self.showQuestionTime == 0:
                    self.currentState = stateGetAnswer
                    self.xMovement = 0
                    self.yMovement = 0

            elif self.currentState == stateGetAnswer:
                img, pt = self.tracker.trackFaces()
                self.videoManager.showQuestion(img, self.questions[self.currentQuestionIndex], self.frameDimensions)
                isAnswer, self.answer, self.xMovement, self.yMovement = self.tracker.getAnswer(self.xMovement, self.yMovement, movementThreshold)
                
                if isAnswer != False:
                    self.currentState = stateShowAnswer
                    self.showAnswerTime = showAnswerMaxTime

            elif self.currentState == stateShowAnswer:
                ret, img = self.cap.read()
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
                ret, img = self.cap.read()

            elif self.currentState == stateDebug:
                if not self.isDebugInit:
                    self.isDebugInit, img, pt = self.tracker.initTracker()
                else:
                    img, pt = self.tracker.trackFaces()
                    isAnswer, self.answer, self.xMovement, self.yMovement = self.tracker.getAnswer(self.xMovement, self.yMovement, debugMovementThreshold)

                    if isAnswer and self.debugShowAnswerTime > 0:
                        self.videoManager.showAnswer(img, self.answer, self.frameDimensions)
                        self.debugShowAnswerTime -= 1

                    if self.debugShowAnswerTime == 0:
                        self.xMovement = 0
                        self.yMovement = 0
                        self.debugShowAnswerTime = debugShowAnswerMaxTime

                    cv.putText(img, 'xMovement: ' + str(self.xMovement), (50, self.frameDimensions[1] - 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv.putText(img, 'yMovement: ' + str(self.yMovement), (50, self.frameDimensions[1] - 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
            cv.imshow('Q&A with Face Detection', img)

            ch = cv.waitKey(1)
            if ch == 27:
                break
