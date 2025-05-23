import torch
from transformers import ASTModel
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tqdm
from dataset.data_generator import DataGenerator
from pedalboard import Gain, Distortion, PitchShift, LowpassFilter, HighpassFilter
import json
import pandas as pd
from model.classifier import EffectClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

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
effects = [Gain,Distortion, PitchShift, HighpassFilter, LowpassFilter]

generator = DataGenerator(effects_to_parameters, effects,parameters=False)


try:
    with open('data/nsynth-train.jsonwav/nsynth-train/examples.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError as e:
    print("You need to download the NSynth dataset first, or change the path to the examples.json file.")
    raise e
df = pd.DataFrame.from_records(data)
df = df.T
guitar_df = df[df['instrument_family_str'] == 'guitar']
elctric_guitar_df = guitar_df[guitar_df['instrument_source_str'] == "electronic"]
elctric_guitar_df = elctric_guitar_df.sample(2000)
dry_tones = [dry_tone + ".wav" for dry_tone in elctric_guitar_df['note_str'].tolist()]
train_data, test_data = train_test_split(dry_tones, test_size=0.2)
test_data, val_data = train_test_split(test_data, test_size=0.5)
print("generating data")
train_dataset = generator.create_data(4, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=train_data,max_chain_length=1)
test_dataset = generator.create_data(4, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=test_data,max_chain_length=1)
val_dataset = generator.create_data(4, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=val_data,max_chain_length=1)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

def eval(model, loss_fn, dl, batch_size = 4):
    model.eval()
    total_loss = 0
    labels = []
    labels_ = []
    preds = []
    logits = []
    for batch in tqdm.tqdm(dl):
        wet_spectrogram = batch['dry_tone']['spectrogram'].to(device)
        dry_spectrogram = batch['wet_tone']['spectrogram'].to(device)
        label = batch['effects'].to(device)
        with torch.no_grad():
            logits_ = model(wet_spectrogram, dry_spectrogram)
        loss = loss_fn(logits_, label)
        total_loss += loss.item()
        for i in range(logits_.shape[0]):
            preds.append(torch.argmax(logits_[i], dim=0).cpu().numpy())
            labels.append(torch.argmax(label[i], dim=0).cpu().numpy())
            labels_.append(torch.nn.functional.one_hot(torch.argmax(label[i], dim=0), num_classes=5).cpu().numpy())
            logits.append(logits_[i].cpu().numpy())
    loss = total_loss / len(dl) * batch_size
    accuracy = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels_, logits)
    print(f"Test: Accuracy:{accuracy} | AUROC: {auroc} | Avg Loss:{loss}")
    return loss, accuracy, auroc

def train(model, optimizer, loss_fn, train_loader,test_loader,lr_scheduler, epochs=10):
    model.train()
    best_accuracy = 0
    labels = []
    labels_ = []
    preds = []
    logits = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            wet_spectrogram = batch['dry_tone']['spectrogram'].to(device)
            dry_spectrogram = batch['wet_tone']['spectrogram'].to(device)
            label = batch['effects'].to(device)
            output = model(wet_spectrogram,dry_spectrogram)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for i in range(output.shape[0]):
                preds.append(torch.argmax(output[i], dim=0).detach().cpu().numpy())
                labels.append(torch.argmax(label[i], dim=0).detach().cpu().numpy())
                labels_.append(torch.nn.functional.one_hot(torch.argmax(label[i], dim=0), num_classes=5).detach().cpu().numpy())
                logits.append(output[i].detach().cpu().numpy())
        print(f"Train: Epoch {epoch+1} | Accuracy: {accuracy_score(labels,preds)} | AUROC: {roc_auc_score(labels_,logits)} | Loss: {total_loss}")
        loss, accuracy, auroc = eval(model, loss_fn, test_loader)
        lr_scheduler.step(loss)
        if accuracy > best_accuracy:
            print(f"saving model at epoch {epoch+1}")
            best_accuracy = accuracy
            #torch.save(model.state_dict(), "saved_models/multiclass_model.pth")
    return

metadata = generator.get_metadata()
import pickle

# Save metadata to pickle file
with open('saved_models/classifier_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)


model = EffectClassifier(5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

print("Beginning model training")
train(model, optimizer, loss_fn, train_loader, test_loader,scheduler, epochs=20)
print("Model training complete")
print("Evaluating model")
eval(model, loss_fn, val_loader)
