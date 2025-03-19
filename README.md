Tone Grabber 🎸🎛️
Tone Grabber is a machine learning project aimed at helping musicians recreate the tone of their favorite artists or recordings by predicting the audio effects needed to transform a dry input tone into a target tone. By leveraging advanced neural network architectures and audio processing techniques, Tone Grabber models analyze differences between two audio signals and recommend specific effects (e.g., reverb, delay, gain) along with their parameters to achieve the desired transformation.

## Features
Currently we have built out multiclass classifiers based on

## Installation

To get started with Tone Grabber, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/tone-grabber.git
    cd tone-grabber
    ```

2. Install the required dependencies:
    Linux/Mac:
    ```sh
    python3 -m venv tone-grabber-venv
    source tone-grabber-venv/bin/activate
    pip install -r requirements.txt
    ```
    Windows:
    ```powershell
    python -m venv tone-grabber-venv
    tone-grabber-venv\Scripts\activate
    pip install -r requirements.txt    
    ```
## Usage

To replicate experiments with the multiclass classifier, first install the nsynth dataset into the data folder from <a here href=https://magenta.tensorflow.org/datasets/nsynth>

Then run:
```sh
python train_classifier.py
```

To replicate the parameter prediction experiments, install the nsynth dataset if you haven't already then run:
```sh
python train_parameter_models.py
```

To run the web based UI locally, once you have either replacated the experiments or downloaded the model weights from <a here href="https://drive.google.com/drive/folders/1zrtoVf5tIh8cRU3_MX_TqXIOkF0kpJ1r?usp=sharing">:
```sh
flask run
```
To get a more in depth understanding of how the code works, check out the demo notebook `demo.ipynb`