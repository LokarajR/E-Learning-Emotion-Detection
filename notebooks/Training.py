# ==============================================================================
# ## Part A: Setup and Imports
# ==============================================================================
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")


# ==============================================================================
# ## Part B: Define Local Paths
# ==============================================================================

BASE_DRIVE_PATH = "/path/to/your/DAiSEE"


DATASET_PATH = os.path.join(BASE_DRIVE_PATH, "DataSet")
LABELS_PATH = os.path.join(BASE_DRIVE_PATH, "Labels")
OUTPUT_PATH = os.path.join(BASE_DRIVE_PATH, "Output")
os.makedirs(OUTPUT_PATH, exist_ok=True)
FRAMES_DIR_PATH = os.path.join(OUTPUT_PATH, "frames")
os.makedirs(FRAMES_DIR_PATH, exist_ok=True)


if not (os.path.exists(DATASET_PATH) and os.path.exists(LABELS_PATH)):
    print("ERROR: Could not find 'DataSet' or 'Labels' folders. Please double-check your BASE_DRIVE_PATH.")
else:
    print("SUCCESS: Found the 'DataSet' and 'Labels' folders.")


# ==============================================================================
# ## Part C: Balance the Dataset Files (Undersampling)
# ==============================================================================
print("\n--- Starting Dataset Balancing Script ---")
LABEL_COLUMNS = ['Boredom', 'Engagement', 'Confusion',]

all_labels_df = pd.read_csv(os.path.join(LABELS_PATH, "AllLabels.csv"))
clip_to_split = {}
for split in ["Train", "Test", "Validation"]:
    with open(os.path.join(DATASET_PATH, f"{split}.txt"), 'r') as f:
        for clip_id in f:
            clip_to_split[clip_id.strip()] = split
master_df_original = all_labels_df.copy()
master_df_original['split'] = master_df_original['ClipID'].map(clip_to_split)
master_df_original.dropna(subset=['split'], inplace=True)


master_df_original['dominant_label'] = master_df_original[LABEL_COLUMNS].idxmax(axis=1)


train_df_original = master_df_original[master_df_original['split'] == 'Train'].copy()
print("\nOriginal training set distribution:")
print(train_df_original['dominant_label'].value_counts())
n_minority = train_df_original['dominant_label'].value_counts().min()
print(f"\nRarest class has {n_minority} samples. Undersampling other classes to match.")
balanced_train_df = train_df_original.groupby('dominant_label').apply(lambda x: x.sample(n_minority, random_state=42)).reset_index(drop=True)


print(f"\nWriting new balanced files to: {OUTPUT_PATH}")
new_train_clips = balanced_train_df['ClipID'].tolist()
with open(os.path.join(OUTPUT_PATH, 'Train_balanced.txt'), 'w') as f:
    for clip in new_train_clips: f.write(f"{clip}\n")
print(f"Created Train_balanced.txt with {len(new_train_clips)} samples.")

for split in ["Test", "Validation"]:
    clips_to_write = master_df_original[master_df_original['split'] == split]['ClipID'].tolist()
    with open(os.path.join(OUTPUT_PATH, f'{split}_balanced.txt'), 'w') as f:
        for clip in clips_to_write: f.write(f"{clip}\n")
    print(f"Created {split}_balanced.txt with {len(clips_to_write)} samples.")
print("\n--- Balancing Script Finished ---\n")


# ==============================================================================
# ## Part D: Load the NEW Balanced Data Information
# ==============================================================================
def create_clip_path_map(dataset_path):
    clip_map = {}
    for split in ["Train", "Test", "Validation"]:
        for root, dirs, files in os.walk(os.path.join(dataset_path, split)):
            for file in files:
                if file.endswith((".avi", ".mp4")):
                    clip_map[file] = os.path.join(root, file)
    return clip_map

def load_balanced_data_info(clip_path_map):
    all_data = []
    all_labels_df = pd.read_csv(os.path.join(LABELS_PATH, "AllLabels.csv"))
    clip_to_split_balanced = {}

    for split in ["Train", "Test", "Validation"]:
        with open(os.path.join(OUTPUT_PATH, f'{split}_balanced.txt'), 'r') as f:
            for clip_id in f: clip_to_split_balanced[clip_id.strip()] = split

    for _, row in all_labels_df.iterrows():
        clip_id = row['ClipID']
        if clip_id in clip_path_map and clip_id in clip_to_split_balanced:
            entry = {'ClipID': clip_id, 'video_path': clip_path_map[clip_id], 'frames_path': os.path.join(FRAMES_DIR_PATH, clip_id.split('.')[0]), 'split': clip_to_split_balanced[clip_id]}
            entry.update(row[LABEL_COLUMNS].to_dict())
            all_data.append(entry)
    master_df = pd.DataFrame(all_data)
    print(f"Loaded information for {len(master_df)} videos from balanced split files.")
    return master_df

clip_to_path_map = create_clip_path_map(DATASET_PATH)
master_df = load_balanced_data_info(clip_to_path_map)


# ==============================================================================
# ## Part E: Frame Extraction (Full Dataset)
# ==============================================================================
SEQUENCE_LENGTH, IMG_SIZE = 30, 224
def extract_frames(video_path, output_folder):
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) == SEQUENCE_LENGTH: return
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < SEQUENCE_LENGTH: return
    frame_indices = np.linspace(0, frame_count - 1, SEQUENCE_LENGTH, dtype=int)
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(output_folder, f"frame_{i:04d}.jpg"), frame)
    cap.release()

print(f"\n--- IMPORTANT: Starting frame extraction for the ENTIRE dataset ({len(master_df)} videos). This will take a long time. ---")
for _, row in tqdm(master_df.iterrows(), total=len(master_df), desc="Extracting All Frames"):
    extract_frames(row['video_path'], row['frames_path'])
print("Full frame extraction complete.\n")


# ==============================================================================
# ## Part F: PyTorch Dataset, Model, and Training
# ==============================================================================
class DAiSEEDataset(Dataset):
    def __init__(self, df, transform): self.df, self.transform = df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        labels = torch.tensor(row[LABEL_COLUMNS].values.astype(np.float32), dtype=torch.float32)
        frames = []
        if os.path.exists(row['frames_path']) and len(os.listdir(row['frames_path'])) == SEQUENCE_LENGTH:
            for i in range(SEQUENCE_LENGTH):
                frame = cv2.imread(os.path.join(row['frames_path'], f"frame_{i:04d}.jpg"))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.transform and frames: frames = torch.stack([self.transform(frame) for frame in frames])
        return frames, labels

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTM, self).__init__()
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        for param in self.feature_extractor.parameters(): param.requires_grad = False
        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        b, s, c, h, w = x.shape; x = x.view(b * s, c, h, w)
        f = self.feature_extractor(x).mean([2, 3]).view(b, s, -1)
        lo, _ = self.lstm(f); o = self.fc1(lo[:, -1, :])
        return self.fc2(self.relu(o))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_df = master_df[master_df['split'] == 'Train']
val_df = master_df[master_df['split'] == 'Validation']
BATCH_SIZE = 8 # Increased batch size slightly for full run

train_dataset = DAiSEEDataset(train_df, transform=transform)
val_dataset = DAiSEEDataset(val_df, transform=transform)

# Since the dataset is now balanced, we use a standard loader with shuffle=True
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNNLSTM(num_classes=len(LABEL_COLUMNS)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 200 # Increased epochs slightly for full run

print(f"\nStarting training on the full balanced dataset ({len(train_df)} samples)...")
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        if len(inputs) == 0: continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs)
        loss = criterion(outputs, labels); loss.backward(); optimizer.step()

        # Clear the CUDA cache after each batch
        torch.cuda.empty_cache()
        gc.collect()


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if len(inputs) == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    print(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "daisee_pytorch_model_balanced.pth"))
print("\nModel training complete.")