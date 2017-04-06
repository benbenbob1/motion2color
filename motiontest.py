import datetime
import time
import imutils
import numpy as np
# and now the most important of all
import cv2

MIN_AREA = 500 #minimum area size, pixels
VIDEO_FEED_SIZE = 500 #pixels
G_BLUR_AMOUNT = 21 #gaussian blur value
DIFF_THRESH = 25 #difference threshold value
LEARN_TIME = 10 #number of identical frames needed to learn the background

numFramesIdentical = 0 #increases every nearly identical frame 
# TODO: use this ^
# get video feed from default camera device
camera = cv2.VideoCapture(0)

print("Attaching to camera...")

while (True):
    if not camera.isOpened():
        time.sleep(2)
    else:
        break

print("Video feed opened")
avgFrame = None

dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

person1Size = 0
person2Size = 0

while (True):
    # get single frame
    response, frame = camera.read()
    if not response:
        print("Error: could not obtain frame")
        # couldn't obtain a frame
        break

    text = "No movement"
    # resize frame
    frame = imutils.resize(frame, width=VIDEO_FEED_SIZE)
    # convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur it to reduce noise
    gray = cv2.GaussianBlur(gray, (G_BLUR_AMOUNT, G_BLUR_AMOUNT), 0)

    if avgFrame is None:
        print "Collecting background info..."
        avgFrame = gray.copy().astype("float")
        continue

    # accumulate average frame
    cv2.accumulateWeighted(gray, avgFrame, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avgFrame))
    threshold = cv2.threshold(frameDelta, DIFF_THRESH, 255, 
        cv2.THRESH_BINARY)[1]

    # dilate - this fills in gaps
    threshold = cv2.dilate(threshold, dilateKernel, iterations=8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, dilateKernel)
    _, contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)

    if person1Size > MIN_AREA:
        person1Size -= 10000
    if person2Size > MIN_AREA:
        person2Size -= 10000



    if len(contours) > 0:
        text = "Movement detected"
        for cont in contours:
            contourArea = cv2.contourArea(cont)

            # ignore areas smaller than MIN_AREA
            if (contourArea < MIN_AREA or
                contourArea < person1Size or
                contourArea < person2Size):
                continue

            if (person1Size > person2Size):
                person2Size = person1Size

            np.average(cont)

            person1Size = contourArea

            text += " ["+str(contourArea)+"]"
            (x,y,w,h) = cv2.boundingRect(cont)
            
            matrixRect = frame[x:(x+w), y:(y+h)]
            avgCols = np.uint8(
                np.average(
                    np.average(matrixRect, axis=0),
                axis=0)
            )
            #print str(avgCols[0]) +", "+str(avgCols[1])+", "+str(avgCols[2])

            cv2.rectangle(
                frame, 
                (x,y), (x+w, y+h), 
                (
                    int(avgCols[0]),
                    int(avgCols[1]),
                    int(avgCols[2])
                ), 
                thickness=5)
    else:
        person1Size = 0
        person2Size = 0

    dateTimeStr = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")

    # putText(frame,text,origin,font_face,font_scale,color,thickness)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 
        0.5, (255,0,0), 1)
    cv2.putText(frame, dateTimeStr, (10, 30), cv2.FONT_HERSHEY_PLAIN, 
        0.5, (255,255,255), 1)
    cv2.putText(frame, "Press q to quit", (10, 40), cv2.FONT_HERSHEY_PLAIN, 
        0.5, (255,255,255), 1)

    cv2.imshow("Feed", frame)
    cv2.imshow("Avg", avgFrame)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Delta", frameDelta)

    # exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
print("Video feed closed")
cv2.destroyAllWindows()