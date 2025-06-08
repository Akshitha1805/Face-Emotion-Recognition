from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np


# Load the pre-trained Keras model
model = load_model('emotion_model.h5', compile=False)
# Compile model manually with correct optimizer argument
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Emotion labels - modify if your model uses different order or names
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (model expects grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)

        # Preprocess face for the model
        face_resized = cv2.resize(face, (64, 64))
        face_normalized = face_resized / 255.0  # Normalize pixel values
        face_input = face_normalized.reshape(1, 64, 64, 1)  # Add batch and channel dims

        # Predict emotion
        prediction = model.predict(face_input)
        max_prob = np.max(prediction)


        emotion_index = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_index]

        # Draw rectangle around face and put emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    # Show the frame with detections and predictions
    cv2.imshow('Facial Emotion Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

