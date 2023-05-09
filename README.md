# Face Analysis and Recognition 📷

This project allows you to analyze and recognize faces using a webcam or uploaded photos using streamlit. The project utilizes the OpenCV, Streamlit, face_recognition, and DeepFace libraries to detect faces, analyze emotions, and recognize individuals.

## Installation 💻

1. Clone the repository:

git clone https://github.com/vilhess/ ...


2. Install the required libraries:

pip install -r requirements.txt

## Usage 🎥

1. To use the webcam feature, run the following command:

streamlit run live_recognition.py


2. To upload a photo and analyze it, navigate to the 'Upload' tab on the left-hand side of the screen.

3. Follow the instructions to take a photo and enter your name. Press 'Enter' and wait for the information to appear. Then, click 'Run' to view the results.

## Notes 📝

- Only one photo per person should be taken, as the system can only detect facial expressions. However, if you are recording a video, the system can detect multiple people if individual frames are saved.

- Be sure to delete any previous recognitions at the end. 🗑️

- The system may not always recognize faces accurately, so it is important to verify the results manually. 🔍




