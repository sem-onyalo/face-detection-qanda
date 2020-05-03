import json
import argparse
import cv2 as cv
from facetracker import FaceTracker
from videomanager import VideoManager
from qanda import App, stateInit, stateDebug

def getAppQuestions():
    try:
        with open('app.settings.json', 'r') as file:
            settings = json.loads(file.read())
            return settings['questions']
    except:
        return [
            'Is this a cool app?',
            'Was it difficult to build?',
            'Should it be shared with others?'
        ] # debugging sample questions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resol", type=str, default="640,480", help="The resolution of the video. Default is '640,480'")
    parser.add_argument("-m", "--mode", type=int, default=0, help="mode to run the app in")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose debugging")
    args = parser.parse_args()

    frameDim = list(map(int, args.resol.split(',')))

    cap = cv.VideoCapture(0)
    faceClassifier = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faceTracker = FaceTracker(cap, frameDim, faceClassifier)
    videoManager = VideoManager()

    appQuestions = getAppQuestions()
    
    if args.mode == 0: # regular
        app = App(cap, faceTracker, videoManager, stateInit, appQuestions, frameDim)
        app.run()
    elif args.mode == 1: # debug app
        app = App(cap, faceTracker, videoManager, stateDebug, appQuestions, frameDim, args.verbose)
        app.run()
    elif args.mode == 2: # debug face tracker
        faceTracker.debug()
 