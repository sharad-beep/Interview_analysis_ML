from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Load the face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Open the video capture
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Create a window for displaying the camera feed
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

def generate_frames():
    while True:
    # Read a frame from the camera
        ret, frame = cap.read()

        # Perform face analysis using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'])

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the dominant emotion on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, result[0]['dominant_emotion'],
                    (0, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
