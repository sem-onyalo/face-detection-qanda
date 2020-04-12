import cv2 as cv

class VideoManager:
    def formatText(self, text, textFont, textScale, textThickness, frameDim, padding):
        textObj = []
        lineCount = 1
        size = cv.getTextSize(text, textFont, textScale, textThickness)
        width = size[0][0]
        height = size[0][1]
        while width > frameDim[0]:
            lineCount += 1
            splitPoint = text[:int(len(text)/lineCount)].rfind(' ')
            splitText = text[:splitPoint]
            size = cv.getTextSize(splitText, textFont, textScale, textThickness)
            width = size[0][0]
            height = size[0][1]

        start = 0
        end = int(len(text)/lineCount)
        topPadding = padding[0]
        bottomPadding = padding[2]
        linePadding = 10
        currentLine = 1
        while True:
            splitPoint = -1 if end >= len(text) else text[start:end].rfind(' ')
            splitText = text[start:end][:splitPoint] if splitPoint > 0 else text[start:]
            size = cv.getTextSize(splitText, textFont, textScale, textThickness)
            width = size[0][0]
            height = size[0][1]
            xPos = int(round((frameDim[0] / 1 - width) / 2))

            if topPadding > 0:
                yPos = topPadding + height
            elif bottomPadding > 0:
                yPos = frameDim[1] - bottomPadding

            textObj.append({ 'text': splitText, 'pt': (xPos,yPos) })
            
            if splitPoint < 0:
                break

            currentLine += 1
            start = start + splitPoint + 1
            end = currentLine * int(len(text)/lineCount)

            if topPadding > 0:
                topPadding += height + linePadding
            elif bottomPadding > 0:
                bottomPadding -= height - linePadding
            
            if currentLine >= 7: # arbitrary cutoff
                break

        return textObj

    def addText(self, img, text, frameDim, textColor=(238,238,238), textScale=1, textThickness=2, textPadding=[0,0,0,0]):
        font = cv.FONT_HERSHEY_SIMPLEX
        textSize = cv.getTextSize(text, font, textScale, textThickness)
        textWidth = textSize[0][0]
        textHeight = textSize[0][1]
        textObjs = self.formatText(text, font, textScale, textThickness, frameDim, textPadding)
        
        for textObj in textObjs:
            cv.putText(img, textObj['text'], textObj['pt'], 
                    font, textScale, textColor, textThickness, cv.LINE_AA)

    def showQuestion(self, img, text, frameDim):
        padding = [20,0,0,0]
        self.addText(img, text, frameDim, (0,0,255), textPadding=padding)

    def showAnswer(self, img, text, frameDim):
        padding = [0,0,20,0]
        self.addText(img, text, frameDim, (0,0,255), textPadding=padding)
