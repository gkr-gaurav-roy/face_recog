import os
import cv2
import face_recognition
import numpy as np

# Replace with your actual directory
directory = "C:\\Users\\GAURAV KUMAR RAY\\OneDrive\\Desktop\\face\\img"

# Create a list to store the known face encodings and names
known_faces = []
known_face_names = []

# Iterate through all images
for file in os.listdir(directory):
    if file.endswith((".jpg", ".jpeg", ".png")):  # Filter for image files
        image_path = os.path.join(directory, file)
        print(f"Processing image: {image_path}")

        # Load the image
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) > 0:
            # Extract the face encoding from the image
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                # Extract the name from the file name
                name = file.split('_')[0]

                # Add the face encoding and name to the lists
                known_faces.append(face_encoding)
                known_face_names.append(name)
            else:
                print(f"No face encodings found for image: {image_path}")
        else:
            print(f"No face locations found for image: {image_path}")

print("Images loaded and face encodings generated!")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding to the known face encodings
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Data not found"

        # If a match is found, get the name
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw a box around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the output
    cv2.imshow('Face Recognition', frame)

    # Press 'x' to exit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
