import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def predict(model, test_dl_aug, device):
    print("> Testing")
    start = time.time() #time generation

    labels_actual = []
    labels_predicted = []

    threshold = 0.88
    i = 0
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl_aug:
            # Move data to the appropriate device (e.g., GPU)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            probability = torch.sigmoid(outputs)

            predicted = (probability > threshold).long()
            predicted = predicted.squeeze()
             
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            labels_actual.extend(labels.cpu().numpy())
            labels_predicted.extend(predicted.cpu().numpy())

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    end = time.time()

    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    cm = confusion_matrix(labels_actual, labels_predicted)
    # show confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('END')

    return 100 * correct / total


def thresholding(model, device, test_dl_aug):
    """
    Evaluate model accuracy across different threshold values."""
    accuracy = []
    opt_thres = 0.5
    for i in range(25, 50):
        print("> Testing: threshold = ", i)
        start = time.time() #time generation

        threshold = i*0.02

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dl_aug:
                # Move data to the appropriate device (e.g., GPU)
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # _, predicted = torch.max(outputs.data, 1)
                # predicted = (outputs.squeeze(1) > 0).long()
                probability = torch.sigmoid(outputs)
                
                predicted = (probability > threshold).long()
                predicted = predicted.squeeze()
                # predicted = torch.round(torch.sigmoid(outputs))
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if (100 *correct / total) > max(accuracy):
            opt_thres = threshold
        accuracy.append(100 * correct / total)
        
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        
    # plot accuracy
    plt.plot(accuracy)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.show()
    print('END')

    return opt_thres, max(accuracy)