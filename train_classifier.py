import torch
from transformers import ASTModel
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tqdm
from dataset.data_generator import DataGenerator
from pedalboard import Chorus, Reverb, Delay, Distortion, Gain
import json
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


class EffectClassifier(torch.nn.Module):
    def __init__(self, n_classes, embed_dim=768):
        super(EffectClassifier, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 1764, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=2, dropout=.1, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.cls = torch.nn.Linear(embed_dim, n_classes)
    def forward(self, x_wet, x_dry):
        x_wet = self.cnn(x_wet.unsqueeze(1))  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry.unsqueeze(1))  # Adjust unsqueeze dimension
        x_wet = self.mlp(x_wet)
        x_dry = self.mlp(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)
        x, _ = self.attn(x, x, x)  # Unpack attn output
        x = self.cls(self.fc(x))
        return x

effects_parameters = {
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
    "Gain": {
        "gain_db": (-60, 24)
    },
    "Chorus": {
        "rate_hz": (0.1, 5.0),
        "depth": (0, 1),
        "centre_delay_ms": (0, 50),
        "feedback": (-1, 1),
        "mix": (0, 1)
    },
    "Distortion": {
        "drive_db": (0, 60)
    }
    }

effects = [Chorus, Reverb, Delay, Gain, Distortion]

generator = DataGenerator(effects_parameters, effects)


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
elctric_guitar_df = elctric_guitar_df.sample(1000)
dry_tones = [dry_tone + ".wav" for dry_tone in elctric_guitar_df['note_str'].tolist()]
train_data, test_data = train_test_split(dry_tones, test_size=0.2)
test_data, val_data = train_test_split(test_data, test_size=0.5)
print("generating data")
train_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=train_data,max_chain_length=1)
test_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=test_data,max_chain_length=1)
val_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=val_data,max_chain_length=1)
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
        wet_features = batch['wet_tone_features'].to(device)
        dry_features = batch['dry_tone_features'].to(device)
        label = batch['effects'].to(device)
        with torch.no_grad():
            logits_ = model(wet_features, dry_features)
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
            wet_features = batch['wet_tone_features'].to(device)
            dry_features = batch['dry_tone_features'].to(device)
            label = batch['effects'].to(device)
            output = model(wet_features,dry_features)
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
            torch.save(model.state_dict(), "saved_models/multiclass_model.pth")
    return

model = EffectClassifier(5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.000002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.25)
loss_fn = torch.nn.CrossEntropyLoss()

print("Beginning model training")
train(model, optimizer, loss_fn, train_loader, test_loader,scheduler, epochs=20)
print("Model training complete")
print("Evaluating model")
eval(model, loss_fn, val_loader)