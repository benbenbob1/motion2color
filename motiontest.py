import datetime
import time
import imutils
import warnings
import numpy as np
# and now the most important of all
import cv2

VIDEO_FEED_SIZE = [272, 204] #[width, height] in pixels
MIN_AREA = (VIDEO_FEED_SIZE[0]*VIDEO_FEED_SIZE[1])/25 #minimum area size, pixels
G_BLUR_AMOUNT = 13 #gaussian blur value
DIFF_THRESH = 50 #difference threshold value
LEARN_APPROVE = 15 #allowed difference between 'identical' frames
LEARN_TIME = 50 #number of identical frames needed to learn the background
FPS = 6

COLOR_SPREAD = 5 # number of margin leds before + after the colorbar to light up
FADE_AMT_PER_FRAME = 0.1 * 255 # amount to fade between every frame


camera = None
piCapture = None
useDisplay = True

isPi = False
try:
    import picamera as pc
    from picamera.array import PiRGBArray
    isPi = True
except ImportError:
    isPi = False

opcLED = True
ledController = None

if opcLED:
    import opc
else:
    import apa102

#METHODS

warnings.simplefilter("ignore")

person1Size = 0
person2Size = 0
numFramesIdentical = 0 #increases every nearly identical frame 
lastFrame = None
numLeds = 0

leds = None

#[[r,g,b], [r,g,b], ...]
def sendLEDs(arr):
    normalized = np.fmin(np.fmax(arr, 0), 255)
    global ledController
    if opcLED:
        ledController.put_pixels(normalized, channel=0)
    else:
        for i in range(numLeds):
            ledController.setPixel(
                numLeds-i, 
                normalized[i][0], normalized[i][1], normalized[i][2]
            )
            ledController.setPixel(
                numLeds+i, 
                normalized[i][0], normalized[i][1], normalized[i][2]
            )
        ledController.show()

def doLoop(isPi):
    global leds
    bgFrame = None

    dilateKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,15))
    closeKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,15))

    leds = np.uint8([[0,0,0]] * numLeds)

    # Returns: 
    # (
    #   bool: should continue loop,
    #   matrix?: background frame or None
    # )
    def processFrame(frame, bgFrame):
        global person1Size, person2Size, numFramesIdentical, lastFrame, leds

        leds = np.fmin(np.fmax(np.subtract(leds,FADE_AMT_PER_FRAME), 0), 255);

        text = "No movement"
        # resize frame
        frame = imutils.resize(frame, 
            width=VIDEO_FEED_SIZE[0]
        )
        # convert it to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur it to reduce noise
        gray = cv2.GaussianBlur(gray, (G_BLUR_AMOUNT, G_BLUR_AMOUNT), 0)

        if (numFramesIdentical >= LEARN_TIME
            or bgFrame is None
            or lastFrame is None):
            shouldUpdate = True
            if (lastFrame is not None and bgFrame is not None):
                bgDelta = cv2.absdiff(bgFrame, lastFrame)
                frameDiffMax = np.uint8(
                    np.max(
                        np.max(bgDelta, axis=0),
                    axis=0)
                )
                if (frameDiffMax <= LEARN_APPROVE):
                    shouldUpdate = False
            if shouldUpdate:
                print "(Re)collecting background frame..."
                lastFrame = gray.copy()
                numFramesIdentical = 0
                return (True, lastFrame)

        # accumulate average frame
        #cv2.accumulateWeighted(gray, avgFrame, 0.5)
        grayDelta = cv2.absdiff(gray, lastFrame)
        frameDelta = cv2.absdiff(gray, bgFrame)
        threshold = cv2.threshold(frameDelta, DIFF_THRESH, 255, 
            cv2.THRESH_BINARY)[1]

        lastFrame = gray

        frameDiffMax = np.uint8(
            np.max(
                np.max(grayDelta, axis=0),
            axis=0)
        )

        if (frameDiffMax <= LEARN_APPROVE):
            numFramesIdentical += 1
        else:
            numFramesIdentical = 0

        # dilate and then close - this fills in gaps
        threshold = cv2.dilate(threshold, dilateKernel, iterations=1)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, closeKernel)
        try:
            _, contours, _ = cv2.findContours(
                threshold.copy(), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        except: #for pi
            _, contours = cv2.findContours(
                threshold.copy(),
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

        if person1Size > MIN_AREA:
            person1Size -= 10000
        if person2Size > MIN_AREA:
            person2Size -= 10000

        justMovement = np.float16(cv2.bitwise_and(frame, frame, mask=threshold))
        justMovement[justMovement == 0] = np.nan

        #TODO
        #leds = np.uint8([[0,0,0]] * numLeds)

        if contours is not None and len(contours) > 0:
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

                person1Size = contourArea
                

                text += " ["+str(contourArea)+"]"
                (x1,y1,w,h) = cv2.boundingRect(cont)
                x2 = min((x1+w), VIDEO_FEED_SIZE[0])
                y2 = min((y1+h), VIDEO_FEED_SIZE[1])
            
                matrixRect = justMovement[
                    y1:y2,
                    x1:x2
                ]
                matrixRect[matrixRect == np.inf] = np.nan

                mean = np.nanmean(
                    np.nanmean(matrixRect, axis=0),
                axis=0)
                if np.isnan(mean).any():
                    continue

                avgCols = np.uint8([[mean]])

                avgColHSV = cv2.cvtColor(avgCols, cv2.COLOR_BGR2HSV)

                avgColHSV[0][0][1] = 255 # Saturation
                avgColHSV[0][0][2] = 255 # Value

                avgCols = cv2.cvtColor(avgColHSV, cv2.COLOR_HSV2BGR)

                cv2.rectangle(
                    frame, 
                    (x1,y1), (x2, y2), 
                    (
                        int(avgCols[0][0][0]),
                        int(avgCols[0][0][1]),
                        int(avgCols[0][0][2])
                    ), 
                    thickness=4)
                #except:
                #    print "Exception drawing box"
                #    print "X: "+str(x)+" Y: "+str(y)+" X+W: "+str(x+w)+" Y+H: "+str(y+h)
                #    continue

                ledStartIdx = (x1 * numLeds) / VIDEO_FEED_SIZE[0]
                ledEndIdx = (x2 * numLeds) / VIDEO_FEED_SIZE[0]
                #print str(ledStartIdx)+" : "+str(ledEndIdx)
                for l in range(1,COLOR_SPREAD):
                    colorAmt = 255 - (l / COLOR_SPREAD) * 255
                    col = [
                        int(avgCols[0][0][2] - colorAmt),
                        int(avgCols[0][0][1] - colorAmt),
                        int(avgCols[0][0][0] - colorAmt)
                    ]
                    thisEnd = min(ledEndIdx+l, numLeds-1)
                    thisStart = max(ledStartIdx-l, 0)
                    leds[thisEnd] += col
                    leds[thisStart] += col
                
                leds[ledStartIdx:ledEndIdx] += [
                    int(avgCols[0][0][2]),
                    int(avgCols[0][0][1]),
                    int(avgCols[0][0][0])
                ]

        circleMargin = 5
        circleRadius = 1
        circleXStart = int(
            (VIDEO_FEED_SIZE[0] / 2.0) - 
            ( (numLeds / 2.0) * (circleMargin + circleRadius) )
        )
        circleY = int(VIDEO_FEED_SIZE[1] / 2.0)
        for c in range(0,numLeds):
            cv2.circle(
                frame,
                (circleXStart, circleY),
                circleRadius,
                (
                    int(leds[c][2]),
                    int(leds[c][1]),
                    int(leds[c][0])
                )
            )
            circleXStart += circleMargin + circleRadius
        else:
            person1Size = 0
            person2Size = 0

        dateTimeStr = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")

        '''
        # putText(frame,text,origin,font_face,font_scale,color,thickness)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 
            0.5, (255,0,0), 1)
        cv2.putText(frame, dateTimeStr, (10, 30), cv2.FONT_HERSHEY_PLAIN, 
            0.5, (255,255,255), 1)
        cv2.putText(frame, "Press q to quit", (10, 40), cv2.FONT_HERSHEY_PLAIN, 
            0.5, (255,255,255), 1)
        '''

        if useDisplay:
            cv2.imshow("Feed", frame)
            cv2.imshow("Movement", np.uint8(justMovement))
        
            #cv2.imshow("Background", bgFrame)
            #cv2.imshow("Threshold", threshold)
            #cv2.imshow("Delta", frameDelta)

            # exit on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return (False, None)

        sendLEDs(leds.tolist())

        return (True, None)

    if (isPi):
        print "Using Pi's PiCamera"
        camera = pc.PiCamera()
        camera.resolution = tuple(VIDEO_FEED_SIZE)
        camera.framerate = FPS
        piCapture = PiRGBArray(camera, size=tuple(VIDEO_FEED_SIZE))
        time.sleep(2.5)
        print("Pi video feed opened")
        for f in camera.capture_continuous(
            piCapture, 
            format="bgr",
            use_video_port=True):
            frame = f.array
            (loop, bg) = processFrame(frame, bgFrame)
            if (not loop):
                piCapture.truncate(0)
                break
            elif (bg is not None):
                piCapture.truncate(0)
                bgFrame = bg
            piCapture.truncate(0)
        closeGently(isPi, None)
    else:
        print "Using CV2's VideoCapture"
        # get video feed from default camera device
        camera = cv2.VideoCapture(0)
        while (True):
            if not camera.isOpened():
                time.sleep(2)
            else:
                break
        print("CV2 video feed opened")   
        while (True):
            # get single frame
            response, frame = camera.read()
            if not response:
                print("Error: could not obtain frame")
                # couldn't obtain a frame
                break

            (loop, bg) = processFrame(frame, bgFrame)
            if (not loop):
                break
            elif (bg is not None):
                bgFrame = bg

        closeGently(isPi, camera)

def closeGently(isPi, camera):
    if (not isPi):
        camera.release()

    print("Video feed closed")
    cv2.destroyAllWindows()

#ENDMETHODS







print("Attaching to camera...")
if opcLED:
    numLeds = 30
    ledController = opc.Client('rpi.student.rit.edu:7890')
    if ledController.can_connect():
        print('Connected to LED OPC')
else:
    numLeds = 180 # 180 * 2 strips
    ledController = apa102.APA102(numLeds*2, 31)

doLoop(isPi)