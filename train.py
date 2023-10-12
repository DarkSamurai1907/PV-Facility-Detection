import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import UNET

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    validate_and_check_accuracy,
    save_predictions_as_imgs,
    test_model
)


# HYPERPARAMETERS
LEARNING_RATE = 0.00001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 50
# NUM_WORKERS = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# PIN_MEMORY = True
LOAD_MODEL = False
DATA_DIR = "imagesv2"
CLIP_VALUE = 5


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    total_loss = 0.0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        optimizer.zero_grad()

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # update from tqdm
        loop.set_postfix(loss=loss.item())

    average_loss = total_loss / len(loader)
    print(f"Training Loss: {average_loss:.4f}")


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=1,
        verbose=True,
        min_lr=1e-8
    )

    early_stopping_patience = 3
    best_val_loss = float('inf')
    patience_counter = 0

    train_loader, val_loader, test_loader = get_loaders(
        DATA_DIR,
        BATCH_SIZE,
        TRAIN_RATIO,
        VAL_RATIO
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        val_loss = validate_and_check_accuracy(val_loader, model, loss_fn, device=DEVICE)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Validation loss has not improved for {early_stopping_patience} epochs. Stopping early.")
            break

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images", device=DEVICE
        )

    save_predictions_as_imgs(
        test_loader, model, folder="test_images", device=DEVICE
    )

    test_model(test_loader, model, device="cuda")


if __name__ == "__main__":
    main()
