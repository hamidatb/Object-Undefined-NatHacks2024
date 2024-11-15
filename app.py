# Apply eventlet monkey-patching FIRST
import eventlet
import signal
import sys

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
from utils.predict_quadrant import QuadrantPredictor
from utils.sample_mood_model import MoodModel
import cv2
import time

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

# Load the mood model
mood_model = load_or_regenerate_model(regenerate=True)

# Initialize Quadrant Predictor
try:
    quadrant_predictor = QuadrantPredictor(model_path='models/look_at_quadrants_model.pkl', scaler_path='models/scaler.pkl')
except FileNotFoundError as e:
    print(e)
    exit()

# Global variables
mood = None
playlist = []
tracking_thread = None
tracking_active = False

# Screen dimensions
# Dynamically get screen resolution using JavaScript and send to server if needed
SCREEN_WIDTH = 1920  # Replace with your actual screen width if known
SCREEN_HEIGHT = 1080  # Replace with your actual screen height if known

# Function to map quadrants to screen coordinates
def map_quadrant_to_screen(quadrant):
    if quadrant == 'top_left':
        return 100, 100  # Adjust based on your preference
    elif quadrant == 'top_right':
        return SCREEN_WIDTH - 100, 100
    elif quadrant == 'bottom_left':
        return 100, SCREEN_HEIGHT - 100
    elif quadrant == 'bottom_right':
        return SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100
    else:
        return SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  # Center

def tracking_function():
    global tracking_active
    tracking_active = True
    cap = cv2.VideoCapture(1)  # Change to 1 if your camera is on index 1
    if not cap.isOpened():
        print("Error: Unable to access camera.")
        tracking_active = False
        return

    print("Starting quadrant tracking...")
    while tracking_active:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Predict the quadrant and get the annotated frame
        quadrant, annotated_frame = quadrant_predictor.predict(frame)

        # If a quadrant is detected, emit the data
        if quadrant:
            screen_x, screen_y = map_quadrant_to_screen(quadrant)
            # Emit gaze coordinates via SocketIO
            socketio.emit('gaze', {'x': screen_x, 'y': screen_y, 'quadrant': quadrant})
            print(f"Gaze Emitted: Quadrant={quadrant}, ScreenX={screen_x}, ScreenY={screen_y}")

        # Optional: Sleep to reduce CPU usage
        time.sleep(0.02)  # Approximately 50 FPS

    # Release resources
    cap.release()
    tracking_active = False
    print("Quadrant tracking stopped.")

def start_tracking():
    global tracking_thread, tracking_active
    if not tracking_active:
        tracking_thread = threading.Thread(target=tracking_function)
        tracking_thread.daemon = True  # Ensure thread exits when main program does
        tracking_thread.start()
        print("Tracking thread started.")
    else:
        print("Tracking is already active.")

# Start tracking automatically when the server starts
start_tracking()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

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
    global tracking_active
    tracking_active = False
    return render_template('thankyou.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


import signal
import sys

# Define a function to handle termination signals
def graceful_exit(signum, frame):
    global tracking_active
    print("\nGracefully shutting down...")

    # Stop tracking if active
    if tracking_active:
        print("Stopping tracking...")
        tracking_active = False
        if tracking_thread and tracking_thread.is_alive():
            tracking_thread.join(timeout=2)  # Set timeout to 2 seconds

    # Stop the Flask-SocketIO server
    socketio.stop()
    print("Server stopped.")
    sys.exit(0)  # Exit the application

# Register the signal handlers
signal.signal(signal.SIGINT, graceful_exit)  # Handles Ctrl+C
signal.signal(signal.SIGTERM, graceful_exit) # Handles termination signal

if __name__ == '__main__':
    socketio.run(app, debug=True)

