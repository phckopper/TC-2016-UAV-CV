# This is a small code capable of tracking objects of a specific color
# Reference: OpenCV Docs. <http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html>. Accessed in March 10th, 2016
# Reference: Learn OpenCV. Blob detection using OpenCV (Python, C++). <http://www.learnopencv.com/blob-detection-using-opencv-python-c> Accessed in March 10th, 2016

import sys

import cv2
import numpy as np

# Open capture device (webcam)
cap = cv2.VideoCapture(1)

# Parses argument and initializes video file if required
if "--save" in sys.argv:
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

# Instantiate a params object so we can configure our blob detector
params = cv2.SimpleBlobDetector_Params()

# Our marks aren't circles, so we disable the circularity filter
params.filterByCircularity = False

# We disable filtering by color because we're already doing this on our own
params.filterByColor = False

# We're already filtering the image using blurs, erosions and dilatations, so we lower the minimum blob area
params.minArea = 1

# Instantiate a blob detector using default configs
detector = cv2.SimpleBlobDetector(params)

while(1):

    # Get a frame
    _, frame = cap.read()

    # Blur frame to reduce noise
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert from BGR (standard OpenCV format) to HSV (easier to work with colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of the blue color in HSV format
    lower_blue = np.array([105, 20, 30])
    upper_blue = np.array([135, 255, 255])

    # Create a mask containing only blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    # Bitwise AND to remove everything from the original frame except for the blue pixels
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Create a black and white version of the resulting image (SimpleBlobDetector takes brightness into account when calculating blobs, but we already
    # have filtered out brightness, so we remove this info
    _, bw_res = cv2.threshold(res[:,:,2], 1, 255, cv2.THRESH_BINARY)

    # Detect blue blobs
    keypoints = detector.detect(bw_res)

    # Draws red circles over detected blobs
    # Crude argument parsing to toggle showing the RGB frame or the resulting BW image
    if "--transparent" in sys.argv:
        blobed_image = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        blobed_image = cv2.drawKeypoints(bw_res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Saves frame
    if "--save" in sys.argv:
        out.write(blobed_image)

    # Show the resulting image
    cv2.imshow('result', blobed_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

if "--save" in sys.argv:
    out.release()

cv2.destroyAllWindows()
