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
    due to cuda dependency issues with the versions of pytorch and tensorflow used in this project we recommend using 2 separate enviornments.
    To install the tensorflow enviornment run the following commands 
    ```sh
    python3 -m venv tone-grabber-venv
    source tone-grabber-venv/bin/activate
    pip install -r requirements.txt
    ```
    To install the pytorch eniornment run the following commands
    ```sh
    python3 -m venv tone-grabber-venv-torch
    source tone-grabber-venv-torch/bin/activate
    pip install -r requirements_torch.txt
    ```
## Usage

To replicate experiments with the multiclass classifier, first install the nsynth dataset into the data folder from: https://magenta.tensorflow.org/datasets/nsynth

Then run:
```sh
python train_classifier.py
```
You could also chose to use a different dataset of audio samples, although you may have to make some other adjustments
