import torch
import time

def predict(model, device, test_dl_aug, class_names):
    """
    """
    start = time.time()
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl_aug:
            # Move data to the appropriate device (e.g., GPU)
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
