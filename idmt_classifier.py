# %%
effect_group_mapping = {
    "chorus": 0,  # Chorus group encoded as 0
    "reverb": 1,   # Reverb group encoded as 1
    "tremolo": 2,
    "overdrive": 3,
    "slapbackdelay": 4,
    "phaser": 5,
    "vibrato": 6,
    "distortion": 7,
    "feedbackdelay": 8,
    "flanger": 9
    }


def pair_dry_with_effects(dry_folder, effect_folders, output_json):
    pairs = []
    for dry_file in os.listdir(dry_folder):
        if not dry_file.endswith(".wav"):
            continue

        # Extract the unique identifier from the dry file
        match = re.match(r"(G\d+-\d+)-1111-\d+\.wav", dry_file)
        if not match:
            continue
        identifier = match.group(1)

        for effect_group, folder in effect_folders.items():
            for wet_file in os.listdir(folder):
                if wet_file.startswith(identifier) and wet_file.endswith(".wav"):
                    # Extract effect setting
                    effect_setting_match = re.search(r"-\d{3}(\d)-", wet_file)  # Adjust regex
                    print(f"Processing wet_file: {wet_file}, matched setting: {effect_setting_match.group(1) if effect_setting_match else None}")

                    if effect_setting_match:
                        effect_setting = int(effect_setting_match.group(1)) - 1  # 0-based indexing
                        pairs.append({
                            "dry_path": os.path.join(dry_folder, dry_file),
                            "wet_path": os.path.join(folder, wet_file),
                            "effect_group": effect_group_mapping[effect_group],
                            "effect_setting": effect_setting
                        })

    with open(output_json, "w") as f:
        json.dump(pairs, f, indent=4)

    print(f"Saved {len(pairs)} pairs to {output_json}")


# %%
import os
import re
import json

dry_folder = "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/NoFX"
effect_folders = {
    "chorus": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/Chorus",
    "reverb": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/Reverb",
    "tremolo": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/Tremolo",
    "overdrive": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/Overdrive",
    "slapbackdelay": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/SlapbackDelay",
    "phaser": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/Phaser",
    "vibrato": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon2/Samples/Vibrato",
    "distortion": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/Distortion",
    "feedbackdelay": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/FeedbackDelay",
    "flanger": "/Users/russellgeorge/Downloads/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/Flanger"
}
output_json = "/Users/russellgeorge/Documents/Data/pairs.json"

pair_dry_with_effects(dry_folder, effect_folders, output_json)


# %%
import librosa
import numpy as np

def generate_mel_spectrogram(audio_path, sr=16000, n_mels=128, hop_length=512):
    # Load the audio
    audio, sr = librosa.load(audio_path, sr=sr)
    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def preprocess_and_save_features(pairs_json, output_folder):
    # Load pairs
    with open(pairs_json, "r") as f:
        pairs = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    # Process each pair
    for pair in pairs:
        # Generate and save Mel-spectrograms
        for key in ["dry_path", "wet_path"]:
            audio_path = pair[key]
            mel_spec = generate_mel_spectrogram(audio_path)
            save_path = os.path.join(output_folder, os.path.basename(audio_path) + "_mel.npy")
            np.save(save_path, mel_spec)

    print(f"Saved features to {output_folder}")


# %%
preprocess_and_save_features(
    pairs_json="/Users/russellgeorge/Documents/Data/pairs.json",
    output_folder="/Users/russellgeorge/Documents/Data/Features"
)

# %%
import json
import numpy as np

def load_data_with_groups(metadata_path, features_folder):
    with open(metadata_path, "r") as f:
        data = json.load(f)

    dry_features, wet_features = [], []
    effect_groups, effect_settings = [], []

    for entry in data:
        dry_path = f"{features_folder}/{entry['dry_path'].split('/')[-1]}_mel.npy"
        wet_path = f"{features_folder}/{entry['wet_path'].split('/')[-1]}_mel.npy"
        dry_features.append(np.load(dry_path))
        wet_features.append(np.load(wet_path))
        effect_groups.append(entry["effect_group"])  # 0 or 1
        effect_settings.append(entry["effect_setting"])  # Convert to 0-based indexing

    return (
        np.array(dry_features),
        np.array(wet_features),
        np.array(effect_groups),
        np.array(effect_settings),
    )

# Example usage
dry_features, wet_features, effect_groups, effect_settings = load_data_with_groups(
    "/Users/russellgeorge/Documents/Data/pairs.json", "/Users/russellgeorge/Documents/Data/Features"
)


# %%
print(effect_groups[1], effect_settings[1])

# %%
from sklearn.model_selection import train_test_split


# Split the data into train and test sets
X_train_dry, X_test_dry, X_train_wet, X_test_wet, y_group_train, y_group_test, y_setting_train, y_setting_test = train_test_split(
    dry_features, wet_features, effect_groups, effect_settings, test_size=0.2, random_state=42
)

# Further split the training set into training and validation sets
X_train_dry, X_val_dry, X_train_wet, X_val_wet, y_group_train, y_group_val, y_setting_train, y_setting_val = train_test_split(
    X_train_dry, X_train_wet, y_group_train, y_setting_train, test_size=0.1, random_state=42
)


# %%
import tensorflow as tf
from tensorflow.keras import layers

def build_multi_output_model(input_shape):
    # Input layers for dry and wet features
    input_dry = layers.Input(shape=input_shape, name="dry_input")
    input_wet = layers.Input(shape=input_shape, name="wet_input")

    shared_cnn = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])

    # Process dry and wet features
    dry_features = shared_cnn(input_dry)
    wet_features = shared_cnn(input_wet)

    # Combine
    combined = layers.Concatenate()([dry_features, wet_features])

    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dense(64, activation="relu")(x)

    # Output for effect group
    group_output = layers.Dense(10, activation="softmax", name="group_output")(x)

    # Output for effect setting
    setting_output = layers.Dense(3, activation="softmax", name="setting_output")(x)

    model = tf.keras.Model(inputs=[input_dry, input_wet], outputs=[group_output, setting_output])
    return model


# %%
# Expand dimensions to add the channel dimension
X_train_dry = np.expand_dims(X_train_dry, axis=-1)
X_train_wet = np.expand_dims(X_train_wet, axis=-1)
X_val_dry = np.expand_dims(X_val_dry, axis=-1)
X_val_wet = np.expand_dims(X_val_wet, axis=-1)
X_test_dry = np.expand_dims(X_test_dry, axis=-1)
X_test_wet = np.expand_dims(X_test_wet, axis=-1)

print("X_train_dry shape:", X_train_dry.shape)


# %%
print("Unique effect groups:", np.unique(y_group_train))
print("Unique effect settings:", np.unique(y_setting_train))


# %%
model = build_multi_output_model((128, 63, 1))
model.compile(
    optimizer="adam",
    loss={
        "group_output": "sparse_categorical_crossentropy",
        "setting_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "group_output": "accuracy",
        "setting_output": "accuracy"
    }
)


from tensorflow.keras.utils import to_categorical

history = model.fit(
    [X_train_dry, X_train_wet],
    {"group_output": y_group_train, "setting_output": y_setting_train},
    validation_data=(
        [X_val_dry, X_val_wet],
        {"group_output": y_group_val, "setting_output": y_setting_val}
    ),
    epochs=50,
    batch_size=32
)





# %%
test_loss = model.evaluate(
    [X_test_dry, X_test_wet],
    {"group_output": y_group_test, "setting_output": y_setting_test}
)
print("Test Loss and Accuracy:", test_loss)


# %%
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Group predictions
group_preds = np.argmax(model.predict([X_test_dry, X_test_wet])[0], axis=1)

# Effect group confusion matrix
cm = confusion_matrix(y_group_test, group_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=effect_group_mapping.keys(), yticklabels=effect_group_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Effect Group")
plt.show()

# Classification report
print(classification_report(y_group_test, group_preds, target_names=list(effect_group_mapping.keys())))



