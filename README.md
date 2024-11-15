# Object-Undefined-NatHacks2024
NatHacks2024 Team Project

Sure! Below is the complete code for the NeuroTune web application as per your design document. I'll provide the GitHub repository structure, all the code files, and instructions on how to run the application.

---

## Repository Structure

```
NeuroTune/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── mood.html
│   ├── playlist.html
│   └── thankyou.html
└── models/
    └── mood_model.pkl
```

---

## Code Files

### 1. `app.py`

```python
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import dlib
import threading
import time
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import pandas as pd
import pickle

app = Flask(__name__)

# Spotify API credentials (you need to provide your own)
SPOTIPY_CLIENT_ID = 'YOUR_CLIENT_ID'
SPOTIPY_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'
SCOPE = 'user-read-playback-state,user-modify-playback-state'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=SCOPE))

# Load the pre-trained mood detection model
with open('models/mood_model.pkl', 'rb') as f:
    mood_model = pickle.load(f)

# Global variables
mood = None
playlist = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_session():
    # Here you can add code to initialize Muse S connection and eye tracking
    return redirect(url_for('detect_mood'))

@app.route('/detect_mood')
def detect_mood():
    global mood
    # For Testing Mode, we'll use sample data
    sample_data = pd.read_csv('data/sample_eeg_data.csv')
    # Simulate mood prediction
    mood = mood_model.predict(sample_data)[0]
    return render_template('mood.html', mood=mood.capitalize())

@app.route('/playlist')
def show_playlist():
    global mood, playlist
    # Get playlist from Spotify based on mood
    if mood:
        results = sp.search(q=mood, type='playlist', limit=1)
        playlist_id = results['playlists']['items'][0]['id']
        tracks = sp.playlist_tracks(playlist_id)['items']
        playlist = [{'name': track['track']['name'], 'uri': track['track']['uri']} for track in tracks[:3]]
    return render_template('playlist.html', playlist=playlist)

@app.route('/play_song', methods=['POST'])
def play_song():
    uri = request.form['uri']
    sp.start_playback(uris=[uri])
    return '', 204

@app.route('/thankyou')
def thank_you():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. `requirements.txt`

```
Flask==2.0.1
opencv-python==4.5.3.56
dlib==19.22.0
spotipy==2.19.0
numpy==1.21.1
pandas==1.3.1
scikit-learn==0.24.2
```

### 3. `README.md`

```markdown
# NeuroTune

NeuroTune is a web application that uses EEG data from the Muse S device and eye-tracking technology to create personalized music recommendations.

## Features

- Testing Mode with sample EEG data
- Mood detection using a pre-trained ML model
- Spotify API integration for playlist recommendations
- Eye-tracking for minimal physical interaction

## How to Run

### Prerequisites

- Python 3.7 or higher
- A Spotify Developer account with Client ID and Client Secret

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/NeuroTune.git
   cd NeuroTune
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up Spotify API credentials:

   - Create an application on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
   - Replace `YOUR_CLIENT_ID` and `YOUR_CLIENT_SECRET` in `app.py` with your credentials.

4. Run the application:

   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`.

## Notes

- The eye-tracking and Muse S integration are simulated in Testing Mode.
- Ensure your webcam is accessible for eye-tracking features.

## License

MIT License


### Prerequisites

- Python 3.7 or higher
- A Spotify Developer account (to obtain `CLIENT_ID` and `CLIENT_SECRET`)
- Webcam access for eye-tracking features (optional in Testing Mode)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/NeuroTune.git
   cd NeuroTune
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Spotify API Credentials**

   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
   - Create a new application.
   - Add `http://localhost:5000/callback` to the Redirect URIs in the settings.
   - Replace `YOUR_CLIENT_ID` and `YOUR_CLIENT_SECRET` in `app.py` with your actual credentials.