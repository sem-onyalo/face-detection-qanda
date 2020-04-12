import argparse
import cv2 as cv
from facetracker import FaceTracker
from videomanager import VideoManager
from qanda import App, stateInit, stateDebug

testQuestions = [
    'Is this a cool app?',
    'Was it difficult to build?',
    'Should it be shared with others?'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resol", type=str, default="640,480", help="The resolution of the video. Default is '640,480'")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    args = parser.parse_args()

    frameDim = list(map(int, args.resol.split(',')))
    initState = stateDebug if args.debug else stateInit

    cap = cv.VideoCapture(0)
    faceClassifier = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faceTracker = FaceTracker(cap, faceClassifier)
    videoManager = VideoManager()

    app = App(cap, faceTracker, videoManager, initState, testQuestions, frameDim)

    app.run()
 