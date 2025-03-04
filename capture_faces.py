import cv2
import os
import csv

# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# Load or create user database
csv_file = "users.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Name"])  # Create headers if the file doesn't exist

# Get user details
user_name = input("Enter Your Name: ")
user_id = str(len(os.listdir(dataset_path)) + 1)  # Assign a unique ID

# Save the user in CSV file
with open(csv_file, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([user_id, user_name])

# Create user folder
save_path = os.path.join(dataset_path, user_id)
if os.path.exists(save_path):
    for file in os.listdir(save_path):
        os.remove(os.path.join(save_path, file))  # Delete old images
else:
    os.makedirs(save_path)

cap = cv2.VideoCapture(0)
best_image = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        best_image = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capturing Face", frame)

    if best_image is not None:
        cv2.imwrite(f"{save_path}/face.jpg", best_image)  # Save only one clear image
        break

cap.release()
cv2.destroyAllWindows()
print(f"Face image saved successfully for {user_name}!")
