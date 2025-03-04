# Face Recognition System

## Overview
This project is a simple face recognition system using OpenCV. It consists of three main components:
1. **Face Capture** - Captures a user's face and stores it in a dataset.
2. **Model Training** - Trains a model using LBPH (Local Binary Patterns Histograms) for face recognition.
3. **Face Recognition** - Recognizes faces in real-time using a webcam.

## Features
- Captures and stores user face images.
- Trains a face recognition model.
- Recognizes faces in real-time and displays names.
- Uses OpenCV's LBPH algorithm for facial recognition.
- Stores user details in a CSV file.

## Project Structure
```
├── capture_faces.py     # Script to capture user face images
├── train_model.py       # Script to train the face recognition model
├── recognize_faces.py   # Script to recognize faces in real-time
├── dataset/             # Directory storing face images
├── users.csv            # CSV file storing user ID and names
├── face_model.yml       # Trained model file
└── haarcascade_frontalface_default.xml  # Pretrained Haar Cascade model for face detection
```

## Setup Instructions
### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install opencv-python numpy
```

### 2. Run Face Capture
Capture images of a new user:
```bash
python capture_faces.py
```
This will prompt you to enter your name and capture your face using the webcam.

### 3. Train the Model
After capturing face images, train the model:
```bash
python train_model.py
```
This will generate `face_model.yml` for face recognition.

### 4. Recognize Faces
Once the model is trained, run the face recognition script:
```bash
python recognize_faces.py
```
This will start the webcam and recognize faces in real-time.

## Notes
- The dataset directory (`dataset/`) stores face images categorized by user ID.
- The `users.csv` file stores the mapping between user ID and name.
- The model is saved as `face_model.yml` and used for real-time recognition.
- Press `q` to exit the recognition window.

## Future Enhancements
- Improve recognition accuracy with deep learning models.
- Add support for multiple image captures per user.
- Implement a GUI for easier interaction.

## License
This project is open-source and free to use. Contributions are welcome!

---
Enjoy building your face recognition system! 🚀

