# This is a small code capable of tracking objects of a specific color
# Reference: OpenCV Docs. <http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html>. Accessed in March 10th, 2016
# Reference: Learn OpenCV. Blob detection using OpenCV (Python, C++). <http://www.learnopencv.com/blob-detection-using-opencv-python-c> Accessed in March 10th, 2016

import sys
from math import sqrt
from time import time

import cv2
import numpy as np

TIMEOUT = 0.75

last_fix = 0
time_to_fix = 0
last_pos = (None, None)
locked = False

def is_rectangle(contour):
    # Get minimum rectangle needed to cover the contour
    # For a perfect rectangle it should be the rectangle itself
    rect = cv2.minAreaRect(contour)

    # Calculate the minRect area
    rectArea = rect[1][0] * rect[1][1]

    # Contour area
    cntArea = cv2.contourArea(contour)

    # Calculate ratio so we can filter
    ratio = cntArea/rectArea

    print ratio

    return ratio > 0.50

def distance_to(contour1, contour2):
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)

    center1 = int(M1['m10']/M1['m00']), int(M1['m01']/M1['m00'])
    center2 = int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00'])

    distance = sqrt(abs(center1[0] - center2[0]) ** 2 + abs(center1[1] - center2[1]) ** 2)

    print distance

# Open capture device (webcam)
cap = cv2.VideoCapture(1)

# Parses argument and initializes video file if required
if "--save" in sys.argv:
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while(1):

    # Get a frame
    _, frame = cap.read()

    # Blur frame to reduce noise
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert from BGR (standard OpenCV format) to HSV (easier to work with colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of the blue color in HSV format
    lower_blue = np.array([110, 50, 70])
    upper_blue = np.array([130, 255, 255])

    # Create a mask containing only blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise AND to remove everything from the original frame except for the blue pixels
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Create a black and white version of the resulting image, as required by findContours
    _, bw_res = cv2.threshold(res[:,:,2], 1, 255, cv2.THRESH_BINARY)

    # Find blue contours
    cnts, _ = cv2.findContours(bw_res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Debug messages are always nice
    print "Found ", len(cnts), " contours!"

    # Keep only the seven biggest contours
    # Only filter if there are more than seven items in array
    if len(cnts) > 10:
        cnts = sorted(cnts, key=cv2.contourArea)[-10:]

    # Keep only rectangular contours
    cnts = filter(is_rectangle, cnts)

    # Cap to 5
    if len(cnts) > 6:
        cnts = sorted(cnts, key=cv2.contourArea)[-5:]

    # Draws red circles over detected blobs
    cv2.drawContours(frame, cnts, -1, (0,0,255), 3)

    pos = []
    # Draw locations
    for cnt in cnts:

        # Extract data from contour
        M = cv2.moments(cnt)

        # Calculate center
        center_x = int(M['m10']/M['m00'])
        center_y = int(M['m01']/M['m00'])

        pos.append((center_x, center_y))

        # Format text
        text = "x: {} y: {} area: {}".format(center_x, center_y, M['m00'])

        # Add text to image
        cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # Draw a circle in the middle of the cube
    if len(cnts) > 4:

        a = sorted(pos, key=lambda x: x[0])[2]
        b = sorted(pos, key=lambda x: x[1])[2]

        if a == b:
            last_pos = a
            last_fix = time()
 

    time_to_fix = time() - last_fix
    text = "TIME TO FIX: {:.4f}".format(time_to_fix)
    cv2.putText(frame, text, (250, 50), cv2.FONT_HERSHEY_PLAIN, 2, (50, 100, 255), 2)

    if time_to_fix < TIMEOUT:
        cv2.circle(frame, last_pos, 10, (0, 255, 0), -1)
        text = "x: {} y: {}".format(*last_pos)
        locked = True
    else:
        locked = False

    if locked:
        cv2.putText(frame, "LOCKED", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "UNLOCKED", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Saves frame
    if "--save" in sys.argv:
        out.write(frame)

    # Show the resulting image
    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

if "--save" in sys.argv:
    out.release()

cv2.destroyAllWindows()
