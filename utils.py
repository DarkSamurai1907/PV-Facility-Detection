import os
import torch
import torch.nn as nn
import torchvision
import shutil
import tqdm as tqdm
from dataset import PVDetectionDataset, image_transform, mask_transform
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(data_dir, batch_size, train_ratio, val_ratio):
    annotations_file = "imagesv2\\annotations.json"
    image_folder = "imagesv2"

    dataset = PVDetectionDataset(annotations_file, image_folder, image_transform, mask_transform)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader


def validate_and_check_accuracy(loader, model, loss_fn, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            val_loss = loss_fn(preds, y)
            total_loss += val_loss.item()

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(y)

            intersection = (preds * y).sum().item()
            union = preds.sum().item() + y.sum().item()
            dice_score += (2 * intersection) / (union + 1e-8)

    accuracy = (num_correct / num_pixels) * 100
    dice_score /= len(loader)
    average_loss = total_loss / len(loader)

    print(f"Got {num_correct}/{num_pixels} correct.")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {dice_score:.4f}")
    print("Validation Loss: {:.4f}".format(average_loss))

    model.train()

    return average_loss


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    model.train()


def test_model(test_loader, model, device="cuda"):
    model.eval()  #
    total_correct = 0
    num_pixels = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = torch.sigmoid(model(inputs))

            preds = (outputs > 0.5).float()
            num_pixels += torch.numel(targets)

            TP += ((preds == 1) & (targets == 1)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()

        accuracy = (TP + TN) / num_pixels
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        mcc = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5 if ((TP + FP) * (TP + FN) * (
                    TN + FP) * (TN + FN)) != 0 else 0.0

        total_agreement = (TP + TN) / num_pixels
        expected_agreement = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (num_pixels * num_pixels)
        kappa = (total_agreement - expected_agreement) / (1 - expected_agreement) if (1 - expected_agreement) != 0 else 0.0

        print()
        print("--------------------------------------")
        print("Evaluation Metrics\n")
        print(f"True Positives (TP): {TP}")
        print(f"True Negatives (TN): {TN}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc}")
        print(f"Kappa Coefficient: {kappa}")