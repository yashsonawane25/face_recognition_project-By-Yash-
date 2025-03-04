import cv2
import numpy as np
import os
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load user names from CSV
csv_file = "users.csv"
name_dict = {}
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        name_dict[int(row[0])] = row[1]  # Store ID → Name mapping

# Function to load images and labels
def get_images_and_labels(dataset_path):
    images, labels = [], []
    for person_id in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, person_id, "face.jpg")
        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(int(person_id))
    return images, np.array(labels)

dataset_path = "dataset"
images, labels = get_images_and_labels(dataset_path)

# Train the model
recognizer.train(images, labels)
recognizer.save("face_model.yml")  # Save trained model
print("Training Complete! Model saved as 'face_model.yml'.")

