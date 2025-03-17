from flask import Flask, render_template, request, jsonify
import os
from flask import Flask, render_template, request, flash, session
from flask_session import Session
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import uuid
from pedalboard.io import ReadableAudioFile
import io
import numpy as np
from model.parameter_prediction import ParameterPrediction
from dataset.feature_extractor_torch import FeatureExtractorTorch
from pedalboard import Reverb, Delay, Chorus, Distortion, Gain
from dataset.data_generator import DataGenerator
import torch
import base64
import soundfile as sf  # Changed from 'wavfile' to 'sf' for clarity
from io import BytesIO

app = Flask(__name__)
app.debug = True
#define the base directory
basedir = os.path.abspath(os.path.dirname(__file__))
#initialize dropzone
dropzone = Dropzone(app)
SAMPLE_RATE = 16000
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = ParameterPrediction(num_effects, num_parameters, param_mask, batch_size=1, num_heads=8).to(device)
feature_extractor = FeatureExtractorTorch()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def resample_tone(file):
    with ReadableAudioFile(io.BytesIO(file.read())) as f:
        re_sampled = f.resampled_to(SAMPLE_RATE)
        tone = np.squeeze(re_sampled.read(int(SAMPLE_RATE * f.duration)), axis=0)
        re_sampled.close()
        f.close()
    return tone

def process_audio_from_outputs(effect, params, sample_rate, metadata):
    dry_tone = np.array(session['dry_tone'])
    effect_idx = torch.argmax(effect).item()
    predicted_effect_pb = metadata['effects'][effect_idx]
    params = params.detach().cpu()
    
    softmax = torch.nn.Softmax(dim=0)
    params = softmax(params[0])
    params = params.numpy()
    
    effect_name = metadata['index_to_effect'][effect_idx]
    param_ranges = metadata['effects_to_parameters'][effect_name]
    
    predicted_params = []
    non_zero_params = [p for p in params if p != 0]
    
    for param_val, (param_name, (min_val, max_val)) in zip(non_zero_params, param_ranges.items()):
        true_val = min_val + param_val * (max_val - min_val)
        predicted_params.append(true_val)
    param_names = param_ranges.keys()
    matched_params = {param_name: value for param_name, value in zip(param_names, predicted_params)}
    predicted_effect_with_params = predicted_effect_pb(**matched_params)
    predicted_wet = predicted_effect_with_params(dry_tone, sample_rate)
    return effect_name, matched_params, predicted_wet

def audio_to_base64(audio_np, sample_rate):
    """Convert numpy audio array to base64-encoded WAV string."""
    buffer = BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

@app.route('/upload_wet_tone', methods=['GET', 'POST'])
def upload_wet_tone():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        session['wet_tone'] = tone.tolist()
        return jsonify({"message": "Wet tone uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_dry_tone', methods=['GET', 'POST'])
def upload_dry_tone():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        session['dry_tone'] = tone.tolist()
        return jsonify({"message": "Dry tone uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    """Runs the model on the uploaded dry and wet tones"""
    try:
        if "dry_tone" not in session or "wet_tone" not in session:
            return jsonify({"error": "Both dry and wet tones must be uploaded first"}), 400

        dry_tone = feature_extractor.get_spectrogram(np.array(session["dry_tone"])).to(device)
        wet_tone = feature_extractor.get_spectrogram(np.array(session["wet_tone"])).to(device)
        model_output = model(dry_tone, wet_tone)
        print(f"Model output length: {len(model_output)}")  # Debug print

        # Handle varying model output lengths
        if len(model_output) == 3:
            _, effect, params = model_output  # Unpack as (loss, effect, params)
        elif len(model_output) == 2:
            effect, params = model_output    # Unpack as (effect, params)
        else:
            raise ValueError(f"Unexpected number of elements in model output: {len(model_output)}")

        effect_name, matched_params, predicted_wet = process_audio_from_outputs(effect, params, SAMPLE_RATE, metadata)
        formatted_params = {
            str(key): round(float(value), 3) 
            for key, value in matched_params.items()
        }

        # Convert audio to base64
        dry_tone_np = np.array(session['dry_tone'])  # Retrieve dry tone from session
        predicted_wet_np = predicted_wet             # Predicted wet tone from processing

        dry_tone_base64 = audio_to_base64(dry_tone_np, SAMPLE_RATE)
        predicted_wet_base64 = audio_to_base64(predicted_wet_np, SAMPLE_RATE)

        # Include audio data in the response
        return jsonify({
            "message": "Prediction successful",
            "predicted_effect": effect_name,
            "predicted_parameters": formatted_params,
            "dry_tone": dry_tone_base64,
            "predicted_wet_tone": predicted_wet_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)