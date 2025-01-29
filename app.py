from flask import Flask, request, jsonify
import os
from flask import Flask, render_template, request, flash, session
from flask_session import Session
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import uuid
from pedalboard.io import ReadableAudioFile
import io
import numpy as np
from model.parameter_prediction import ParameterPrediction, process_audio_from_outputs
from dataset.feature_extractor_torch import FeatureExtractorTorch
from pedalboard import Reverb, Delay, Chorus, Distortion, Gain
from dataset.data_generator import DataGenerator

app = Flask(__name__)
#define the base directory
basedir = os.path.abspath(os.path.dirname(__file__))
#initialize dropzone
dropzone = Dropzone(app)
SAMPLE_RATE = 16000
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "supersecretkey"
Session(app)

#TODO: Change this to be read in by the user, ie let the user pick and choose what effects they have start with a checklist + button

effects_to_parameters = {
    "Reverb": {
        "room_size": (0, 1),
        "damping": (0, 1), 
        "wet_level": (0, 1),
        "dry_level": (0, 1),
        "width": (0, 1),
        "freeze_mode": (0, 1)
    },
    "Delay": {
        "delay_seconds": (0, 2),
        "feedback": (0, 1),
        "mix": (0, 1)
    },
    "Chorus": {
        "rate_hz": (0, 100),
        "depth": (0, 1),
        "centre_delay_ms": (1, 30),
        "feedback": (0, 1),
        "mix": (0, 1)
    },
    "Distortion": {
        "drive_db": (0, 100)
    },
    "Gain": {
        "gain_db": (-12, 12)
    }
}
effects = [Reverb, Delay, Distortion, Gain, Chorus]


generator = DataGenerator(effects_to_parameters, effects)
metadata = generator.get_metadata()
param_mask = metadata['parameter_mask_idx']
num_parameters = metadata['total_parameters']
num_effects = len(metadata['effect_to_idx'].keys())
model = ParameterPrediction(num_effects,num_parameters,param_mask,num_heads=8)


@app.route('/')
def index():
    return render_template('index.html')
    
def resample_tone(file):
    with ReadableAudioFile(io.BytesIO(file.read())) as f:
        re_sampled = f.resampled_to(SAMPLE_RATE)
        tone = np.squeeze(re_sampled.read(int(SAMPLE_RATE * f.duration)),axis=0)
        re_sampled.close()
        f.close()
    return tone

@app.route('/upload_wet_tone')
def upload_wet_tone():
    try:
        # Ensure a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        return tone
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_dry_tone')
def upload_dry_tone():
    try:
        # Ensure a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        return tone
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict(wet_tone, dry_tone):
    #TODO: fill out prediction function
    return