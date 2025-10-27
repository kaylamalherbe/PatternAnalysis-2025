from dataset import data_loader_aug, test_loader_aug, calc_normalization_values
import pickle
import torch
import torch
import time
from modules import ConvNextForImageClassification, evaluate_model
import torch.nn as nn 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


data_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\train"
test_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\test"

def evaluate(model, test_dl_aug):
    print("Evaluate Model using Testing")
    start = time.time() #time generation

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl_aug:
            # Move data to the GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    print('END')


def train(model, num_epochs, learning_rate, criterion, optimizer, scheduler=scheduler, load = False):
    
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_patience = 5
    patience_counter = 0

    if load:
        model.load_state_dict(torch.load('best_model_weights.pth'))

    # Train the model
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(dl_aug):
            # Move data to the appropriate device (e.g., GPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dl_aug)}], TRAIN Loss: {loss.item():.4f}')
            scheduler.step()

        # use validation set to find loss and accuracy
        val_loss, val_acc = evaluate_model(model, val_dl_aug, criterion, device)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  -> Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Check if this is the best model so far based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience if performance improved

            # Save the best model weights found on the validation set
            torch.save(model.state_dict(), 'best_model_weights.pth')

        else:
            patience_counter += 1
            # If the model performance hasn't improved for 'patience_counter' epochs, stop training.
            if patience_counter >= early_stop_patience:
                print(f"\n Early stopping triggered after {epoch+1} epochs.")
                break
    print("\n> Training Finished.")
        
    return model

if __name__ == "__main__":

    mean, std = calc_normalization_values(data_location)
    dl_aug, val_dl_aug = data_loader_aug(data_location, batch_size=64, mean=mean, std=std)
    test_dl_aug = test_loader_aug(test_location, batch_size=64, mean=mean, std=std)

    # save to pickle
    with open('train_loader_aug.pkl', 'wb') as f:
        pickle.dump(dl_aug, f)
    with open('val_loader_aug.pkl', 'wb') as f:
        pickle.dump(val_dl_aug, f)
    with open('test_loader_aug.pkl', 'wb') as f:
        pickle.dump(test_dl_aug, f)

    model = ConvNextForImageClassification(
        in_channels=1,  # Changed from 3 to 1
        stem_features=96,
        depths=[3, 3, 9, 3],
        widths=[96, 192, 384, 768],
        drop_p=0.1,
        num_classes=2 # Changed to 2 for binary classification (AD/NC)
    )
    model = model.to(device)

    learning_rate = 1e-6
    max_lr = 0.1
    num_epochs = 50

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    
    #Piecewise Linear Schedule
    total_step = len(dl_aug)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=total_step, epochs=num_epochs)

    #Evaluate model
    model = train(model, num_epochs=32, learning_rate=learning_rate, criterion=criterion, optimizer=optimizer, scheduler=scheduler, load=False)
    evaluate(model, test_dl_aug)

    """Epoch [50/50], Step [100/236], TRAIN Loss: 0.2710
Epoch [50/50], Step [200/236], TRAIN Loss: 0.2622
Epoch [50/50]
  -> Validation Loss: 0.3041, Validation Accuracy: 0.9376
Training took 2148.47243976593 secs or 35.807873996098834 mins in total

> Testing
> Testing with unaug data
Test Accuracy of the model on the 9000 test images: 71.4 %
Testing took 67.7975766658783 secs or 1.1299596110979715 mins in total
END"""