# mood_model.py
# Sample mood prediction model. This will always return happy

import pickle
from sklearn.dummy import DummyClassifier
import pandas as pd
import os
import os
import pickle

class MoodModel:
    def __init__(self):
        from sklearn.dummy import DummyClassifier
        self.model = DummyClassifier(strategy="most_frequent")
    
    def train_dummy_model(self):
        # Ensure the models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Dummy training data
        X = pd.DataFrame({
            'eeg1': [0.5, 0.6, 0.55],
            'eeg2': [0.6, 0.65, 0.6],
            'eeg3': [0.55, 0.6, 0.58],
            'eeg4': [0.6, 0.65, 0.6],
            'eeg5': [0.65, 0.7, 0.68],
            'eeg6': [0.6, 0.65, 0.6],
            'eeg7': [0.58, 0.6, 0.59],
            'eeg8': [0.6, 0.65, 0.6],
            'eeg9': [0.62, 0.67, 0.63],
            'eeg10': [0.6, 0.65, 0.62]
        })
        y = ['happy', 'happy', 'happy']

        # Fit the dummy model
        self.model.fit(X, y)
        
        # Save the trained model
        model_path = os.path.join('models', 'mood_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
