import torch
from config import Config
from data.dataset import create_data_loaders
from models.sam_finetune import SAMFinetune
from torch.optim import AdamW
from training.train import Trainer

from utils.LinearWarmupCosine import LinearWarmupCosineAnnealingLR


def main():
    config = Config()

    # Data loaders
    train_loader, val_loader = create_data_loaders(
        config.IMAGE_FOLDER, config.MASK_FOLDER, config.BATCH_SIZE, 1024
    )

    # Model
    sam_finetune = SAMFinetune(
        config.MODEL_TYPE, config.SAM_CHECKPOINT, config.DEVICE
    )
    model = sam_finetune.get_model()

    scaler = torch.amp.grad_scaler.GradScaler(device=config.DEVICE)
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.NUM_EPOCHS,
        warmup_start_lr=5e-7,
        eta_min=config.ETA_MIN,
    )

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        scaler,
        config.DEVICE,
        config.LOG_DIR,
        config.MODEL_SAVE_DIR,
    )
    trainer.train(config.NUM_EPOCHS)


if __name__ == "__main__":
    main()
