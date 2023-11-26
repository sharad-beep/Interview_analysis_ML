from flask import Flask, request,render_template, jsonify, send_file
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Function to analyze emotions in a video
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    emotions_data = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i % (frame_rate // 6) == 0:  # Get 30 frames per second
            result = DeepFace.analyze(frame, actions=['emotion'])
            dominant_emotion = result[0]['dominant_emotion']
            emotions_data[dominant_emotion] += 1
    
    total_frames = sum(emotions_data.values())
    emotions_percentage = {emotion: (count / total_frames) * 100 for emotion, count in emotions_data.items()}
    
    # Generate plot
    plt.figure(figsize=(8, 6))
    plt.bar(emotions_percentage.keys(), emotions_percentage.values())
    plt.xlabel('Emotions')
    plt.ylabel('Percentage')
    plt.title('Emotion Percentage Throughout the Video')
    plot_path = 'static/video_emotion_plot.png'  # Path to save the generated plot
    plt.savefig(plot_path)  # Save the plot image
    
    cap.release()
    return plot_path

@app.route('/analyze_video', methods=['POST'])
def analyze_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        video_path = 'static/uploads/' + file.filename
        file.save(video_path)
        plot_path = analyze_video(video_path)
        return send_file(plot_path, mimetype='image/png')
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True,port=9000)
