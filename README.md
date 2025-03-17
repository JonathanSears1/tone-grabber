# Tone Grabber 🎸🎛️
Tone Grabber is a machine learning project aimed at helping musicians recreate the tone of their favorite artists or recordings by predicting the audio effects needed to transform a dry input tone into a target tone. By leveraging advanced neural network architectures and audio processing techniques, Tone Grabber models analyze differences between two audio signals and recommend specific effects (e.g., reverb, delay, gain) along with their parameters to achieve the desired transformation.

## Installation

To get started with Tone Grabber, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/tone-grabber.git
    cd tone-grabber
    ```

2. Install the required dependencies:
    ```sh
    python3 -m venv tone-grabber-venv-torch
    source tone-grabber-venv-torch/bin/activate
    pip install -r requirements.txt
    ```
## Usage
To get a better understanding of the tone grabber repo, check out our demo notebook: [demo.ipynb](demo.ipynb)

The demo notebook covers:
- How to use the dataset generator
- How to use the feature extractor class to get the audio spectrogram, loudness, and fundamental frequency 
- How to run inference with the parameter prrediction model and intepreate the outputs to apply the predicted effects to audio samples

To recreate the parameter prediction experiments:
```
python train.py
```
