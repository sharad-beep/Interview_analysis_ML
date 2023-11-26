import requests
import json
import base64
# Replace with your API key
API_KEY = 'YOUR_API_KEY'

# Replace with your audio file path
audio_file_path = 'audio.flac'

# Read the audio file and encode it in base64
with open(audio_file_path, 'rb') as audio_file:
    audio_content = audio_file.read()
    encoded_audio = base64.b64encode(audio_content).decode('utf-8')
    # encoded_audio = base64.b64encode(audio_content).decode('utf-8')  # Python 3

# API endpoint
url = 'https://proxy.api.deepaffects.com/audio/generic/api/v2/sync/recognise_emotion'

# Request payload
payload = {
    "content": encoded_audio,
    "sampleRate": 8000,
    "encoding": "FLAC",
    "languageCode": "en-US"
}

headers = {
    'Content-Type': 'application/json',
    'apikey': API_KEY
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Emotion recognized:", result['segments'][0]['emotion'])
    print("Start time:", result['segments'][0]['start'])
    print("End time:", result['segments'][0]['end'])
else:
    print("Failed to recognize emotion. Status code:", response.status_code)
