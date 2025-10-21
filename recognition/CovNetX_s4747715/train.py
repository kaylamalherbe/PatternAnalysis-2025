import dataset
import pickle

data_location = r"C:\Users\kayla\Documents\Uni\2025 Sem 2\COMP3710\Final Project\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\train"
validation_location = r"C:\Users\kayla\Documents\Uni\2025 Sem 2\COMP3710\Final Project\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\test"

if __name__ == "__main__":
    dl, val_dl = dataset.data_loader(data_location, batch_size=32)
    test_dl = dataset.test_loader(validation_location, batch_size=32)

    # print size of data_loaders
    print(len(dl))
    print(len(val_dl))
    print(len(test_dl))

    # save to pickle
    with open('train_loader.pkl', 'wb') as f:
        pickle.dump(data_loader, f)
    with open('val_loader.pkl', 'wb') as f:
        pickle.dump(val_dl, f)
    with open('test_loader.pkl', 'wb') as f:
        pickle.dump(test_dl, f)