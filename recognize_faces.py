import cv2
import csv

# Load trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load user names from CSV
csv_file = "users.csv"
name_dict = {}
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        name_dict[int(row[0])] = row[1]  # ID → Name mapping

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 50:
            name = name_dict.get(label, "Unknown")
        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
