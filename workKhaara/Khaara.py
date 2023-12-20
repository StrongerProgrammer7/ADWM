import cv2

# Загрузка каскада для лиц
face_cascade = cv2.CascadeClassifier('../xml/haarcascade_frontalface_default.xml')

url = "1.mp4"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("error read movie")
    exit()
#cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detected Faces', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()