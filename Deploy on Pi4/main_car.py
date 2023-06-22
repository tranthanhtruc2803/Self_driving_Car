import lane_detection_module
from motor_module import Motor
from lane_detection_module import getLaneCurve
#import webcam_module

##################################################
motor = Motor(2, 3, 4, 17, 22, 27)


##################################################

def main():
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    while True:
        success, img = cap.read()
        #cv2.imshow("Result", img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    #img = webcam_module.getImg()
    curveVal = getLaneCurve(img, 1)

    sen = 1.3  # SENSITIVITY
    maxVAl = 0.3  # MAX SPEED
    if curveVal > maxVAl: curveVal = maxVAl
    if curveVal < -maxVAl: curveVal = -maxVAl
    # print(curveVal)
    if curveVal > 0:
        sen = 1.7
        if curveVal < 0.05: curveVal = 0
    else:
        if curveVal > -0.08: curveVal = 0
    motor.move(0.20, -curveVal * sen, 0.05)
    cv2.waitKey(1)


if __name__ == '__main__':
    while True:
        main()