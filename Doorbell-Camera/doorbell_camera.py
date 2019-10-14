#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Source :
# https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd
# https://gist.github.com/ageitgey/84943a12dd0d9f54e90f824b94e4c2a9


import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle


# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []


def save_known_faces() :
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces() :
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass


def register_new_face( face_encoding, face_image ) :
    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })


def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        if datetime.now() - metadata["last_seen"] > timedelta( seconds=5 ) :
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

    return metadata


def main_loop():

    # Get the camera
    video_capture = cv2.VideoCapture(0)
    video_capture.set( cv2.CAP_PROP_FRAME_WIDTH, 1280 )
    video_capture.set( cv2.CAP_PROP_FRAME_HEIGHT, 720 )

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/5 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # See if this face is in our list of known faces.
            metadata = lookup_known_face(face_encoding)

            # If we found the face, label the face with some useful information.
            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = f"Visiteur {int(time_at_door.total_seconds())}s"

            # If this is a brand new face, add it to our list of known faces
            else:
                face_label = "Nouveau visiteur !"

                # Grab the image of the the face from the current frame of video
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                # Add the new face to our known face data
                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        # Draw a box around each face and label each face
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/5 size
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display recent visitor images
        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            # If we have seen this person in the last 10s, draw their image
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=5) and metadata["seen_frames"] > 5:
                # Draw the known face image
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                # Label the image with how many times they have visited
                visits = metadata['seen_count']
                visit_label = f"{visits} visites"
                if visits == 1:
                    visit_label = "1 visite"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors == 1 :
            cv2.putText(frame, "Visiteur reconnu", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        elif number_of_recent_visitors > 1 :
            cv2.putText(frame, "Visiteurs reconnus", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)

	# Wait for key pressed
        key = cv2.waitKey(1) & 0xFF
        # Hit 'Escape' on the keyboard to quit!
        if key == 27 :
            break
        # Hit 's' to save the know faces
        elif key == ord('s') :
            save_known_faces()
        # Hit 'l' to load the know faces
        elif key == ord('l') :
            load_known_faces()
        # Hit 'r' to reset the know faces
#        elif key == ord('r') :
#           global known_face_encodings, known_face_metadata
#            known_face_encodings = []
#            known_face_metadata = []


    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Programme principal
if __name__ == "__main__" :

    load_known_faces()
    main_loop()
