from dataset import data_loader_aug, test_loader_aug, calc_normalization_values
import pickle
import torch
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


data_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\train"
test_location = r"FILE-PATH\ADNI data set for Alzheimer's disease-20251015T011003Z-1-001\AD_NC\test"

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