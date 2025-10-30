from dataset import data_loader_aug, test_loader_aug, calc_normalization_values
import pickle
import torch
import torch
import time
from modules import ConvNext, evaluate_model
from predict import predict, thresholding
import torch.nn as nn 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


data_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\train"
test_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\test"

def evaluate_model(model, dataloader, criterion, device, threshold=0.6):
    """Evaluate the model on the validation set.
    Args:   
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        threshold (float): Threshold for converting logits to binary predictions.
    Returns:
        Tuple[float, float]: Average loss and accuracy on the validation set.
    """
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to the appropriate device (e.g., GPU)
            images, labels = images.to(device), labels.to(device)
            
            labels = labels.unsqueeze(1).float() # Unsqueeze labels to match model output shape and cast to float

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Calculate accuracy (assuming multi-class/binary classification)
            # Use torch.round to get binary predictions from logits
            # predicted = torch.round(torch.sigmoid(outputs))
            probability = torch.sigmoid(outputs)
            predicted = (probability > threshold).long()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # Set the model back to training mode
    model.train()
    return avg_loss, accuracy


def train(model, num_epochs, learning_rate, criterion, optimizer, 
          scheduler, load = False):
    """
    Train the ConvNeXt model with early stopping based on validation loss.
    Args:
        model (nn.Module): The ConvNeXt model to train.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. 
                    Defaults to None.
        load (bool, optional): Whether to load existing model weights. 
                    Defaults to False.
    Returns:
        nn.Module: The trained ConvNeXt model.
    """
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_patience = 5
    patience_counter = 0
    val_max = 0.995
    val_sav_points = [80, 85, 90, 95]
    validation_loss_values = []
    validation_acc_values = []
    learning_rate_values = []

    # Train the model
    start = time.time() #time generation
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(dl_aug):
            # Move data to the appropriate device (e.g., GPU)
            images, labels = images.to(device), labels.to(device)
            # print("images to device")
            labels = labels.unsqueeze(1).float() # Unsqueeze labels to match model output shape and cast to float
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i+1) % 10 == 0:
            #   print(outputs)

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dl_aug)}], TRAIN Loss: {loss.item():.4f}')

            scheduler.step()
            learning_rate_values.append(optimizer.param_groups[0]['lr'])


        # use validation set to find loss and accuracy
        val_loss, val_acc = evaluate_model(model, val_dl_aug, criterion, device)
        validation_loss_values.append(val_loss)
        validation_acc_values.append(val_acc)
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
        if epoch % 10 == 0:
            # intermittent saving to check where model is overfitting
            torch.save(model.state_dict(), 'model_weights_' + str(val_acc) +'.pth')
            print("saved case: ", val_acc)
        if val_acc >= val_max:
            print(f"\n Early stopping triggered after {epoch+1} epochs.")
            break

    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " 
            + str(elapsed/60) + " mins in total")

    print("\n> Training Finished.")

    # plot validation loss vs epochs
    import matplotlib.pyplot as plt
    # Plot validation loss vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(validation_loss_values) + 1), validation_loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epochs')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(validation_acc_values) + 1), validation_acc_values)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Epochs')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rate_values) + 1), learning_rate_values)
    plt.xlabel('step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Step')
    plt.grid(True)
    plt.show()
        
    return model

if __name__ == "__main__":

    # mean, std = calc_normalization_values(data_location)
    dl_aug, val_dl_aug = data_loader_aug(data_location, batch_size=64)
    test_dl_aug = test_loader_aug(test_location, batch_size=64)

    # save to pickle
    with open('train_loader_aug.pkl', 'wb') as f:
        pickle.dump(dl_aug, f)
    with open('val_loader_aug.pkl', 'wb') as f:
        pickle.dump(val_dl_aug, f)
    with open('test_loader_aug.pkl', 'wb') as f:
        pickle.dump(test_dl_aug, f)

    # check_validation_split()
    
    model = ConvNext(
        num_channels=1,  # Changed from 3 to 1
        stem_features=96,
        depths=[3, 3, 9, 1],
        widths=[96, 192, 384, 768],
        dropout_p=0.3,
        drop_path_rate=0.4,
        num_classes=2 # Changed to 2 for binary classification (AD/NC)
    )
    model = model.to(device)

    learning_rate = 1e-5
    max_lr = 1e-3 
    num_epochs = 60

    # set criterion to CrossEntropyLoss for multi-class classification
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    #Piecewise Linear Schedule
    total_step = len(dl_aug)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=total_step, epochs=num_epochs, pct_start=0.3)

    #Evaluate model
    model = train(model, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion, optimizer=optimizer, scheduler=scheduler, load=False)
    performance = predict(model, test_dl_aug)
    opt_thres, performance = thresholding(model, device, test_dl_aug)
