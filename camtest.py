import cv2


print('Opening camera...')
vid = cv2.VideoCapture(0)

W=640
H=480
#vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
vid.set(cv2.CAP_PROP_FRAME_WIDTH, W)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
vid.set(cv2.CAP_PROP_FPS, 30)

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    print(frame)
    if not ret:
        continue

    cv2.imshow('frame', frame)

    # the 'q' button is quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()