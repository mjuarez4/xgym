import cv2

cam = cv2.VideoCapture(6)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

cam.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cam.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
