# Robert Griffin Stober / Kabir Vidyarthi
import os
import random
import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import editdistance
import wandb

# =========================
# DATASET
# =========================
class IAMLinesDataset(Dataset):
    def __init__(self, lines_root, xml_root, transform=None):
        self.lines_root = lines_root
        self.xml_root = xml_root
        self.transform = transform
        self.samples = []

        # Parse all xml files
        for xml_file in os.listdir(xml_root):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(xml_root, xml_file)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for line in root.iter("line"):
                line_id = line.attrib["id"]
                text = line.attrib["text"]

                folder1 = line_id.split("-")[0]
                folder2 = "-".join(line_id.split("-")[:2])

                img_path = os.path.join(
                    lines_root, folder1, folder2, line_id + ".png"
                )

                if os.path.exists(img_path):
                    self.samples.append((img_path, text))

        print("Loaded", len(self.samples), "line samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    images, labels = zip(*batch)
    widths = [img.shape[2] for img in images]
    max_width = max(widths)
    padded_images = [F.pad(img, (0, max_width - img.shape[2])) for img in images]
    images = torch.stack(padded_images)
    return images, labels

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 1024)),  # taller + wide
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# CHARACTER MAPS
# =========================
characters = string.ascii_letters + string.digits + string.punctuation + " "
char_to_idx = {c: i+1 for i, c in enumerate(characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
blank_label = 0

def encode_label(text):
    return torch.tensor([char_to_idx[c] for c in text if c in char_to_idx])

# =========================
# CRNN MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        self.rnn = nn.LSTM(
            input_size=256*8,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# =========================
# CTC DECODING
# =========================
def ctc_greedy_decode(output, blank=0):
    preds = torch.argmax(output, dim=2)
    decoded = []
    for pred in preds:
        prev = blank
        string = []
        for p in pred:
            p = p.item()
            if p != blank and p != prev:
                string.append(idx_to_char.get(p, ""))
            prev = p
        decoded.append("".join(string))
    return decoded

def compute_cer(predictions, ground_truths):
    total_distance = 0
    total_chars = 0
    for pred, gt in zip(predictions, ground_truths):
        total_distance += editdistance.eval(pred, gt)
        total_chars += len(gt)
    return total_distance / total_chars if total_chars > 0 else 0

# =========================
# MAIN FUNCTION
# =========================
def main():
    # Paths (change to your actual desktop paths)
    lines_root = "Desktop/Handwriting Analysis Project/lines"
    xml_root = "Desktop/Handwriting Analysis Project/xml"

    # Dataset + Dataloaders
    dataset = IAMLinesDataset(lines_root, xml_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=len(characters)+1).to(device)
    criterion = nn.CTCLoss(blank=blank_label)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 25

    wandb.login(key="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")
    wandb.init(project="handwriting-analysis-IAM", config={
        "epochs": num_epochs,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "architecture": "IAM-CRNN-CTC",
        "dataset": "IAM"
    })
    wandb.watch(model, log="all", log_freq=100)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0
        itr = 0
        for images, labels in trainloader:
            images = images.to(device)
            targets = [encode_label(label) for label in labels]
            target_lengths = torch.tensor([len(t) for t in targets])
            targets = torch.cat(targets).to(device)
            outputs = model(images).log_softmax(2)
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
            loss = criterion(outputs.permute(1,0,2), targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            itr+=1
        epoch_train_loss = train_loss_total / len(trainloader)

        # Validation
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                targets = [encode_label(label) for label in labels]
                target_lengths = torch.tensor([len(t) for t in targets])
                targets = torch.cat(targets).to(device)
                outputs = model(images).log_softmax(2)
                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
                loss = criterion(outputs.permute(1,0,2), targets, input_lengths, target_lengths)
                val_loss_total += loss.item()
                all_preds.extend(ctc_greedy_decode(outputs))
                all_labels.extend(labels)
        epoch_val_loss = val_loss_total / len(testloader)
        cer = compute_cer(all_preds, all_labels)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | CER: {cer:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": epoch_train_loss,
                   "val_loss": epoch_val_loss, "CER": cer})

if __name__ == "__main__":
    main()