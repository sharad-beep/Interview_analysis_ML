from flask import Flask, render_template, request, jsonify,send_file
import os
from flask_cors import CORS
import matplotlib.pyplot as plt
# Your existing Python code here...
import librosa
import soundfile
import os, glob, pickle
import numpy as np
np.float = float
np.complex=complex
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
print('helo')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype=float)
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
        return result
    
#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def lod_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("G:\\speech_emo_rec\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=lod_data(test_size=0.25)

#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))
print('f')

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

def predict_emotion(file_path):
    # Extract features from the audio file
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    # Reshape the feature for prediction (assuming it's a single sample)
    feature = feature.reshape(1, -1)
    # Predict the emotion using the trained model
    emotion_pred = model.predict(feature)
    
    # Print predicted emotion
    print(f"Predicted Emotion sample : {emotion_pred[0]}")
    return emotion_pred[0] 
    
    # Show dominant emotion plot (you can customize this based on your visualization preferences)
    # For example, you could use matplotlib to plot the dominant emotion or display it as text
    
# Example usage:
# file_path_to_test = "path_to_your_audio_file.wav"
# predict_emotion(file_path_to_test)


# File path to the audio file
file_path_to_test = "calm.wav"

# Call the function to predict emotion
predict_emotion(file_path_to_test)


app = Flask(__name__)
CORS(app)
def generate_bar_plot(file_path):
    # Predict emotion for the current audio file
    emotion = predict_emotion(file_path)
    # Count the occurrences of each predicted emotion (this is an example, replace with your logic)
    emotions_data = {
        'calm': 0,
        'happy': 0,
        'fearful': 0,
        'disgust': 0
    }
    emotions_data[emotion] += 1  # Increment the count for the predicted emotion

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(emotions_data.keys(), emotions_data.values())
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Predicted Emotions for Current Audio File')
    plt.ylim(0, 1)  # Set the y-axis limit based on the count (in this case, 0 or 1)
    
    # Save the plot as an image
    plot_path = 'static/bar_plot.png'
    plt.savefig(plot_path)
    
    return plot_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        emotion = predict_emotion(file_path)
        print("Predicted Emotion inside:", emotion)  # For debugging
        # Generate the bar plot
        print("file path " + file_path)
        plot_path = generate_bar_plot(file_path)
         # Generate the full URL to the plot image
        plot_url = plot_path
        return jsonify({'emotion': emotion, 'plot_path': plot_url})
    

@app.route('/get_plot')
def get_plot():
   return send_file('G:/speech_emo_rec/static/bar_plot.png', mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
