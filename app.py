from flask import Flask, request, jsonify
import os
from flask import Flask, render_template, request, session
from flask_session import Session
from flask_dropzone import Dropzone
import uuid
from pedalboard.io import ReadableAudioFile
import io
import numpy as np
from model.parameter_prediction import ParameterPredictionResNet
from model.classifier import EffectClassifier
from dataset.feature_extractor_torch import FeatureExtractorTorch
from pedalboard import Distortion, Gain, LowpassFilter, HighpassFilter, PitchShift

import torch
import base64
import soundfile as sf  # Changed from 'wavfile' to 'sf' for clarity
from io import BytesIO
import matplotlib.pyplot as plt
import librosa.display



app = Flask(__name__)
app.debug = True
#define the base directory
basedir = os.path.abspath(os.path.dirname(__file__))
#initialize dropzone
dropzone = Dropzone(app)
SAMPLE_RATE = 16000
app.config["SESSION_TYPE"] = "filesystem"
# app.config["SECRET_KEY"] = "supersecretkey"
Session(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO: Change this to be read in by the user, ie let the user pick and choose what effects they have start with a checklist + button

effects = [Distortion, Gain, PitchShift, LowpassFilter, HighpassFilter]
effects_to_parameters = {
        "Gain": {
            "gain_db": [-60, 24]
        },
        "Distortion": {
            "drive_db": [0, 60]
        },
        "PitchShift": {
        "semitones": [-12, 12]
        },
        "HighpassFilter": {
        "cutoff_frequency_hz": [20, 20000]
        },
        "LowpassFilter": {
        "cutoff_frequency_hz": [20, 20000]
        }
    }


import pickle

# Load metadata from pickle file
with open('saved_models/classifier_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
param_mask = metadata['parameter_mask_idx']
num_parameters = metadata['total_parameters']
num_effects = len(metadata['effect_to_idx'].keys())
classifier = EffectClassifier(num_effects,batch_size=1).to(device)
classifier.load_state_dict(torch.load("saved_models/multiclass_model.pth",weights_only=False))
classifier.eval()
feature_extractor = FeatureExtractorTorch()
parameter_model_dict = {}
for effect_name, param_dict in effects_to_parameters.items():
    model = ParameterPredictionResNet(768,len(param_dict.values())).to(device)
    model.load_state_dict(torch.load(f"saved_models/{effect_name}_parameter_prediction.pth",weights_only=False),strict=False)
    parameter_model_dict[effect_name] = model.eval()

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
        tone = np.squeeze(re_sampled.read(int(SAMPLE_RATE * f.duration)),axis=0)
        re_sampled.close()
        f.close()
    return tone


def process_audio_from_outputs(effect, params, metadata,sample_rate=16000):
        dry_tone = np.array(session['dry_tone'])
        predicted_effect_pb = metadata['effects'][int(torch.argmax(effect))]
        predicted_params = [float(param) for param in list(params.detach().squeeze(0)) if param != 0]
        effect_name = metadata['index_to_effect'][int(torch.argmax(effect))]
        param_names = metadata['effects_to_parameters'][effect_name].keys()
        
        # Scale parameters between min and max values
        # scaled_params = []
        # for param_name, param_value in zip(param_names, predicted_params):
        #     min_val, max_val = metadata['effects_to_parameters'][effect_name][param_name]
        #     # Squash between 0 and 1 using sigmoid
        #     squashed = 1 / (1 + np.exp(-param_value))
        #     # Scale to range
        #     scaled_value = min_val + squashed * (max_val - min_val)
        #     scaled_params.append(scaled_value)
            
        matched_params = {param_name:value for param_name,value in zip(param_names,predicted_params)}
        predicted_effect_with_params = predicted_effect_pb(**matched_params)
        
        predicted_wet = predicted_effect_with_params.process(dry_tone,sample_rate)
        return effect_name, matched_params, predicted_wet
    

@app.route('/upload_wet_tone',methods=['GET','POST'])
def upload_wet_tone():
    try:
        # Ensure a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        session['wet_tone'] = tone.tolist()
        return jsonify({"message": "Wet tone uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_dry_tone',methods=['GET','POST'])
def upload_dry_tone():
    try:
        # Ensure a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        tone = resample_tone(file)
        session['dry_tone'] = tone.tolist()
        return jsonify({"message": "Dry tone uploaded successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def audio_to_base64(audio_np, sample_rate):
    """Convert numpy audio array to base64-encoded WAV string."""
    buffer = BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

def create_spectrogram_image(audio_data, sample_rate=16000):
    """Create a spectrogram image from audio data and return it as base64 string."""
    # Create mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Create figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect_db, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/predict', methods=['GET'])
def predict():
    """Runs the model on the uploaded dry and wet tones"""
    try:
        if "dry_tone" not in session or "wet_tone" not in session:
            return jsonify({"error": "Both dry and wet tones must be uploaded first"}), 400

        dry_tone_spec = feature_extractor.get_spectrogram(np.array(session["dry_tone"])).to(device)
        wet_tone_spec = feature_extractor.get_spectrogram(np.array(session["wet_tone"])).to(device)
        
        effect = classifier(dry_tone_spec, wet_tone_spec)
        print(effect)
        effect_idx = torch.argmax(effect)
        effect_name = metadata['index_to_effect'][int(effect_idx)]
        print(effect_name)
        
        joint_spec = torch.cat((dry_tone_spec.unsqueeze(0),wet_tone_spec.unsqueeze(0)),dim=1)
        param_model = parameter_model_dict[effect_name]
        params = param_model(joint_spec.to(device))
        print(params)
        # Example if your model supports this
        effect_name, matched_params, predicted_tone = process_audio_from_outputs(effect,params,metadata)
        formatted_params = {
            str(key): round(float(value), 3) 
            for key, value in matched_params.items()
        }
        # Convert audio to base64
        dry_tone_base64 = audio_to_base64(np.array(session['dry_tone']), SAMPLE_RATE)
        predicted_wet_base64 = audio_to_base64(predicted_tone, SAMPLE_RATE)
        wet_tone_base64 = audio_to_base64(np.array(session['wet_tone']), SAMPLE_RATE)
        
        # Create spectrogram images
        dry_spec_img = create_spectrogram_image(np.array(session['dry_tone']))
        wet_spec_img = create_spectrogram_image(np.array(session['wet_tone']))
        predicted_spec_img = create_spectrogram_image(predicted_tone)
        
        return jsonify({
            "message": "Prediction successful",
            "predicted_effect": effect_name,
            "predicted_parameters": formatted_params,
            "dry_tone": dry_tone_base64,
            "wet_tone": wet_tone_base64,
            "predicted_wet_tone": predicted_wet_base64,
            "spectrograms": {
                "dry": dry_spec_img,
                "wet": wet_spec_img,
                "predicted": predicted_spec_img
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}),500

if __name__ == "main":
    app.run(debug=True)

