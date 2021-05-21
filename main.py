import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
    for x,y,w,h in face:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(gray,scaleFactor=1.9,minNeighbors=18)
        for x,y,w,h in smile:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1)

    if key ==ord('x'):
        break
video.release()
cv2.destroyAllWindows()