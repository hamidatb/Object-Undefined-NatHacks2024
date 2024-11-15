# Apply eventlet monkey-patching FIRST
import eventlet
eventlet.monkey_patch()

# Import other modules AFTER monkey-patching
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from utils.eye_tracking import EyeTracker
from utils.sample_mood_model import MoodModel

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Initialize SocketIO with the app
socketio = SocketIO(app, cors_allowed_origins="*")

# Spotify API credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

SCOPE = 'user-read-playback-state,user-modify-playback-state,playlist-read-private'

# Initialize Spotify client
sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                        client_secret=SPOTIPY_CLIENT_SECRET,
                        redirect_uri=SPOTIPY_REDIRECT_URI,
                        scope=SCOPE)

# Load or regenerate the mood detection model
model_path = os.path.join('models', 'mood_model.pkl')
def load_or_regenerate_model(regenerate=False):
    if regenerate or not os.path.exists(model_path):
        print("Training a new mood model...")
        mood_model_obj = MoodModel()
        mood_model_obj.train_dummy_model()
        with open(model_path, 'rb') as f:
            mood_model = pickle.load(f)
        print("Model successfully trained and saved.")
    else:
        with open(model_path, 'rb') as f:
            mood_model = pickle.load(f)
        print("Loaded existing model.")
    return mood_model

# Load the model
mood_model = load_or_regenerate_model(regenerate=True)

# Initialize Eye Tracker with SocketIO
eye_tracker = EyeTracker(socketio, camera_index=1)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_session():
    threading.Thread(target=eye_tracker.start_tracking).start()
    return redirect(url_for('detect_mood'))

@app.route('/detect_mood', methods=['GET', 'POST'])
def detect_mood():
    global mood
    if request.method == 'POST':
        mode = request.form.get('mode')
        session['mode'] = mode
        if mode == 'live':
            pass  # Live mode placeholder
        else:
            sample_data = pd.read_csv(os.path.join('static', 'data', 'sample_eeg_data.csv'))
            mood = mood_model.predict(sample_data)[0]
        return redirect(url_for('show_mood'))
    return render_template('detect_mood.html')

@app.route('/show_mood')
def show_mood():
    global mood
    if not mood:
        return redirect(url_for('detect_mood'))
    return render_template('mood.html', mood=mood.capitalize())

@app.route('/playlist')
def show_playlist():
    global mood, playlist
    if not mood:
        return redirect(url_for('detect_mood'))
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    sp = spotipy.Spotify(auth=token_info['access_token'])
    results = sp.search(q=mood, type='playlist', limit=5)
    playlists = results['playlists']['items']
    if not playlists:
        return "No playlists found for this mood.", 404
    selected_playlist = playlists[0]
    playlist_id = selected_playlist['id']
    tracks = sp.playlist_tracks(playlist_id)['items']
    playlist = [{'name': track['track']['name'], 'uri': track['track']['uri']} for track in tracks[:5]]
    return render_template('playlist.html', playlist=playlist, mood=mood.capitalize())

@app.route('/play_song', methods=['POST'])
def play_song():
    data = request.get_json()
    uri = data.get('uri')
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        return jsonify({'error': 'Unauthorized'}), 401
    sp = spotipy.Spotify(auth=token_info['access_token'])
    sp.start_playback(uris=[uri])
    return jsonify({'status': 'Playing'}), 200

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('show_playlist'))

@app.route('/thankyou')
def thank_you():
    eye_tracker.stop_tracking()
    return render_template('thankyou.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
