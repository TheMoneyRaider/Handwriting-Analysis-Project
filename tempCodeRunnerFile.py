
            targets = [encode_label(label) for label in labels]
            target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long).to(device)
            targets = torch.cat(targets).to(device)
            outputs = model(images).log_softmax(2)
            input_lengths = torch.full((outputs.size(0),),outputs.size(1),dtype=torch.long).to(device)