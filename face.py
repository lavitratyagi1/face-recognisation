import cv2
import os
import uuid
import csv
from datetime import datetime

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 250, 100, 200, 300

# Load images from the 'images' folder
image_folder = 'images'
known_faces = []
known_names = []

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(face_rect) > 0:
            (x, y, w, h) = face_rect[0]
            roi = gray[y:y+h, x:x+w]
            known_faces.append(roi)
            known_names.append(os.path.splitext(filename)[0])  # Extract name from filename

# Create a CSV file to store user details
csv_file = 'user_details.csv'
csv_header = ['Name', 'Timestamp']
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        if x > roi_x and y > roi_y and x + w < roi_x + roi_width and y + h < roi_y + roi_height:
            roi_gray = gray[y:y+h, x:x+w]

            # Compare the detected face with known faces
            match_found = False
            for i, known_face in enumerate(known_faces):
                result = cv2.matchTemplate(roi_gray, known_face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > 0.8:  # Adjust threshold as needed
                    match_found = True
                    name = known_names[i]
                    break

            if match_found:
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Save the detected unknown face
                unknown_face_path = os.path.join('unknown_faces', f'person_{str(uuid.uuid4())[:8]}.jpg')
                cv2.imwrite(unknown_face_path, roi_gray)
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Add user details to CSV file
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Unknown', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
