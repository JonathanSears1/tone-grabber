from dataset.data_generator import DataGenerator
from pedalboard import Reverb, Delay, Chorus, Distortion, Gain
import torch
from torch.nn import MSELoss, CrossEntropyLoss
import os
from model.parameter_prediction import ParameterPrediction
from model.parameter_prediction import Trainer
from auraloss.freq import MultiResolutionSTFTLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
# List of effects
effects = [Reverb, Delay, Distortion, Gain, Chorus]

# create instance of data generator corresponding to effects
generator = DataGenerator(effects_to_parameters, effects)

num_samples = 4
audio_directory = "/home/jonat/tone-grabber/data/nsynth-train.jsonwav/nsynth-train/audio"
dry_tones = os.listdir(audio_directory)
dry_tones = dry_tones[:1000]
# max_chain_length is the maximum number of effects applied to a sample
train_data, test_data = train_test_split(dry_tones, test_size=0.2)
test_data, val_data = train_test_split(test_data, test_size=0.5)

max_chain_length = 1
#set the model
metadata = generator.get_metadata()
param_mask = metadata['parameter_mask_idx']
num_parameters = metadata['total_parameters']
num_effects = len(metadata['effect_to_idx'].keys())
model = ParameterPrediction(num_effects,num_parameters,param_mask,num_heads=8)

import os
import pickle

# Check if datasets already exist
datasets_exist = all(os.path.exists(f'data/{split}_dataset.pkl') 
                    for split in ['train', 'test', 'val'])

if datasets_exist:
    print("Loading existing datasets from pickle files...")
    # Load existing datasets
    with open('data/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('data/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    with open('data/val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    print("Datasets loaded successfully")
else:
    print("Generating new datasets...")
    train_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=train_data,max_chain_length=1)
    test_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=test_data,max_chain_length=1)
    val_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=val_data,max_chain_length=1)
    print("Saving datasets to pickle files...")
    import pickle
    # Save train dataset
    with open('data/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    # Save test dataset 
    with open('data/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    # Save validation dataset
    with open('data/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    print("Datasets saved successfully")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
loss_fn_effect = CrossEntropyLoss()
loss_fn_params = MSELoss()
optimizer = Adam(model.parameters(),.00001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.25)
trainer = Trainer(model, metadata)
model = model.to(device)
trainer.train(model, train_loader, test_loader, loss_fn_effect,loss_fn_params, optimizer, scheduler, 50)
trainer.eval(model,val_loader,loss_fn_effect,loss_fn_params,0)