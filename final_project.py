# Robert Griffin Stober / Kabir Vidyarthi

# Package imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb
import datetime

# File Imports
from dataset import *
from model import *

# Meta things
from meta_config import *

#work for sunday - split this into a train function, an evaluate function, and a load data function
def main():
    # Hyperparameters
    if USES_HPC: # hpc doesn't like having batch sizes above 8
        batch_size = 8
    else:
        batch_size = 16
    lr = 3e-4
    num_epochs = 15

    # Dataset + Dataloaders
    dataset = IAMLinesDataset(transform=transform)
    # if REDUCED_DATASET:
    #     subset_size = len(dataset) // 4
    #     dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
    print("Loaded dataset")

    # Check if cuda speedup is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running model on", device)

    # Train/Test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    
    num_batches = len(trainloader)
    datapoints_per_epoch = 10 # doesn't actually impact wandb recording, only meant to determine how many (persistient) printouts we have between epochs
    epoch_interval = (num_batches // datapoints_per_epoch) + 1 # lazy round up so there's never an excess record

    # Model
    model = CRNN(num_classes=len(characters)+1).to(device)
    criterion = nn.CTCLoss(blank=blank_label, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if SCHEDULAR:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5
        )

    print("loaded model")

    if WANDB_RECORDING:
        wandb.init(project="handwriting-analysis-IAM", config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "IAM-CRNN-CTC",
            "dataset": "IAM"
        })
        wandb.watch(model, log="all", log_freq=100)

    # Training loop
    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0
        prev_total = 0
        for i, (images, labels) in enumerate(trainloader):
            cur_time = datetime.datetime.now()
            runtime = cur_time - start_time
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


            if (i + 1) % epoch_interval == 0 or i + 1 == num_batches:
                interval_train_loss = (train_loss_total - prev_total) / epoch_interval
                print(f'Epoch [{epoch+1:02}/{num_epochs}] Batch [{i+1:04}/{num_batches}] Train Loss: {interval_train_loss:.3f} Elapsed Time: {runtime}')
                prev_total = train_loss_total
            else:
                print(f'Epoch [{epoch+1:02}/{num_epochs}] Batch [{i+1:04}/{num_batches}] Train Loss: {loss.item():.3f} Elapsed Time: {runtime}', end='\r')

        epoch_train_loss = train_loss_total / num_batches

        # Validation
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        logged_images = None
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
                preds = ctc_greedy_decode(outputs)
                all_preds.extend(preds)
                all_labels.extend(labels)

                if WANDB_RECORDING and logged_images is None:
                    logged_images = []

                    for j in range(min(5,images.size(0))):
                        caption = f"GT: {labels[j]} | Pred: {preds[j]}"
                        logged_images.append(wandb.Image(images[j].cpu(), caption=caption))

        epoch_val_loss = val_loss_total / len(testloader)
        cer = compute_cer(all_preds, all_labels)
        
        if SCHEDULAR:
            scheduler.step()

        if not WANDB_RECORDING:
            for i in range(3):
                print("GT :", all_labels[i])
                print("PR :", all_preds[i])

        print("Unique preds:", torch.unique(torch.argmax(outputs,2)))
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | CER: {cer:.4f}")

        if WANDB_RECORDING:
            wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "CER": cer, "validation_examples": logged_images})
    
    #cleanup
    if WANDB_RECORDING:
        wandb.finish()

    if SAVE_MODEL:
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"{dir_path}/models/{MODEL_FILENAME}_{timestamp}.pth"

        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()