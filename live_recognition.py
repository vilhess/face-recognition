import cv2
import streamlit as st
import face_recognition
import numpy as np
import pickle
from deepface import DeepFace
import os
from pathlib import Path

# Dictionnaires pour la correspondance des émotions et des genres
EMOTION_DICT = {"angry": "😡", "disgust": "🤢", "fear": "😱",
                "happy": "😄", "sad": "😢", "surprise": "😮", "neutral": "😐"}

GENDER_DICT = {"Man": "♂", "Woman": "♀"}


def analyze_image(image_path):
    """Analyser l'image et retourner l'âge, l'émotion et le genre."""
    try:
        res = DeepFace.analyze(image_path)[0]

        age = res["age"]
        emotion = EMOTION_DICT[res["dominant_emotion"]]
        gender = GENDER_DICT[res["dominant_gender"]]

        return age, emotion, gender

    except:
        return None, None, None


def recognize_face(frame, known_face_encodings, known_face_names):
    """Reconnaître les visages dans la frame et les nommer."""
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


def main():
    st.markdown("<h1 style='text-align: center;'>Face analysis and recognition</h1>",
                unsafe_allow_html=True)
    st.subheader("")

    st.markdown("<h5 style='text-align: center;'>To use this feature, simply take a photo and enter your name. Press the 'Enter' key once and wait until the information appears. Then, click 'Run' and observe the results.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Please take one photo per person, as the system can only read facial expressions. However, in the case of a video, the system can detect multiple people if you save individual frames. Please ensure to delete any previous recognitions at the end.</h5>", unsafe_allow_html=True)

    st.subheader("")

    # Charger les encodages de visage connus et les noms
    path1 = "function_cam/known_face_encodings"
    path2 = "function_cam/known_face_names"
    if Path(path1).is_file() and Path(path2).is_file():
        known_face_encodings = pickle.load(
            open(path1, 'rb'))
        known_face_names = pickle.load(
            open(path2, 'rb'))
    else:
        known_face_encodings = []
        known_face_names = []

    if st.checkbox('open camera'):
        picture = st.camera_input("Take a picture")

        if picture is not None:
            # Enregistrer la nouvelle image
            with open("function_cam/test.jpg", "wb") as f:
                f.write(picture.getbuffer())

            # Demander le nom de l'utilisateur
            names = st.text_input("What is your name?")

            st.markdown(
                f"<h2 style='text-align: center;'>Your name is {names}</h1>", unsafe_allow_html=True)

            # Encoder le nouveau visage
            new_pic = face_recognition.load_image_file(
                "function_cam/test.jpg")
            new_pic_encoding = face_recognition.face_encodings(new_pic)[0]

            # Ajouter le nouveau visage aux encodages de visage connus et aux noms
            if names != "":
                known_face_encodings.append(new_pic_encoding)
                known_face_names.append(names)

            # Sauvegarder les encodages de visage connus et les noms
            pickle.dump(known_face_encodings, open(
                "function_cam/known_face_encodings", 'wb'))
            pickle.dump(known_face_names, open(
                "function_cam/known_face_names", 'wb'))

            # Analyser l'image
            age, emotion, gender = analyze_image("function_cam/test.jpg")

            # Afficher les résultats de l'analyse
            if age is not None and emotion is not None and gender is not None:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader(f"age : {age} yo")
                with col2:
                    st.subheader(f"emotion : {emotion}")
                with col3:
                    st.subheader(f"gender : {gender}")

            # Afficher le flux vidéo en direct avec la reconnaissance faciale
            process_this_frame = True

            st.header("")
            st.markdown(
                "<h1 style='text-align: center;'>Webcam Live Feed</h1>", unsafe_allow_html=True)
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)

            while run:
                _, frame = camera.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Reconnaître les visages dans la frame
                if process_this_frame:
                    face_locations, face_names = recognize_face(
                        frame, known_face_encodings, known_face_names)

                process_this_frame = not process_this_frame

                # Nommer les visages dans la frame
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 0, 255), 2)

                    cv2.rectangle(frame, (left, bottom - 35),
                                  (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                font, 1.0, (255, 255, 255), 1)

                # Afficher la frame
                FRAME_WINDOW.image(frame)
            else:
                st.write('')

            # Bouton pour supprimer les reconnaissances précédentes
            st.subheader("")
            st.subheader("")
    if st.button('delete previous recognition'):
        try:
            known_face_encodings = []
            known_face_names = []

            pickle.dump(known_face_encodings, open(
                        "function_cam/known_face_encodings", 'wb'))
            pickle.dump(known_face_names, open(
                        "function_cam/known_face_names", 'wb'))
            os.remove("function_cam/test.jpg")
        except:
            st.write("No previous recognition")


main()
