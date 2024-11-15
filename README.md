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