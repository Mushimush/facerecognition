import cv2
import os

# Ask user for name
name = input("Enter your name: ")

# Create directory with user's name
folder_path = f"/home/stanley/Documents/facerecognition/FR/{name}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the number of images to capture
num_images = 1000
count = 0

# Capture and save 500 images
while count < num_images:
    # Capture frame from camera
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the face image to the directory with the user's name
        img_name = os.path.join(folder_path, str(count) + '.jpg')
        cv2.imwrite(img_name, frame[y:y+h, x:x+w])

        # Increment the count
        count += 1
        print(count)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
