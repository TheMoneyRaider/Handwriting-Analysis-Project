# Robert Griffin Stober / Kabir Vidyarthi

# Package imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb

# File Imports
from dataset import *
from model import *

# =========================
# MAIN FUNCTION
# =========================
def main():
    # Paths (change to your actual desktop paths)
    lines_root = "Desktop/Handwriting Analysis Project/lines"
    xml_root = "Desktop/Handwriting Analysis Project/xml"

    # Dataset + Dataloaders
    dataset = IAMLinesDataset(lines_root, xml_root, transform=transform)

    # -------------------------
    # USE ONLY 1/8 OF DATASET
    # -------------------------
    subset_size = len(dataset) // 8
    dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # Train/Test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=len(characters)+1).to(device)
    criterion = nn.CTCLoss(blank=blank_label)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = 25

    wandb.init(project="handwriting-analysis-IAM", config={ # setup a .netrc file for login information, I deleted your key because I wanted to have my tests on my acc so I could see them instead of having to ask you
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
            target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long).to(device)
            targets = torch.cat(targets).to(device)
            outputs = model(images).log_softmax(2)
            input_lengths = torch.full((outputs.size(0),),outputs.size(1),dtype=torch.long).to(device)
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
                target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long).to(device)
                targets = torch.cat(targets).to(device)
                outputs = model(images).log_softmax(2)
                input_lengths = torch.full((outputs.size(0),),outputs.size(1),dtype=torch.long).to(device)
                loss = criterion(outputs.permute(1,0,2), targets, input_lengths, target_lengths)
                val_loss_total += loss.item()
                all_preds.extend(ctc_greedy_decode(outputs))
                all_labels.extend(labels)
        epoch_val_loss = val_loss_total / len(testloader)
        cer = compute_cer(all_preds, all_labels)
        if epoch % 1 == 0:
            for i in range(3):
                print("GT :", all_labels[i])
                print("PR :", all_preds[i])
        print("Unique preds:", torch.unique(torch.argmax(outputs,2)))
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | CER: {cer:.4f}")
        wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "CER": cer})

if __name__ == "__main__":
    main()