import os

import torch


class Config:
    # Data paths
    IMAGE_FOLDER = "./data/CVC-ClinicDB/PNG/Original"
    MASK_FOLDER = "./data/CVC-ClinicDB/PNG/Ground Truth"
    SAVE_FOLDER = "./output/CVC-ClinicDB/predict"

    # Model parameters
    SAM_CHECKPOINT = "./checkpoints/sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    LEARNING_RATE = 4e-6
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 30
    ETA_MIN = 1e-6

    # Logging
    LOG_DIR = "./logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Model saving
    MODEL_SAVE_DIR = "./output/CVC-ClinicDB/SAM Finetune Enc Dec"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
