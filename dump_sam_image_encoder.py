import os

import torch
from segment_anything import SamPredictor, sam_model_registry

if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints", exist_ok=True)

sam_h_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
sam_l_checkpoint = "./checkpoints/sam_vit_l_0b3195.pth"
sam_b_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"

sam_h = sam_model_registry["vit_h"](checkpoint=sam_h_checkpoint)

sam_l = sam_model_registry["vit_l"](checkpoint=sam_l_checkpoint)

sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_checkpoint)

torch.save(
    sam_h.image_encoder.state_dict(),
    "./checkpoints/sam_vit_h_image_encoder.pth",
)

torch.save(
    sam_l.image_encoder.state_dict(),
    "./checkpoints/sam_vit_l_image_encoder.pth",
)

torch.save(
    sam_b.image_encoder.state_dict(),
    "./checkpoints/sam_vit_b_image_encoder.pth",
)

print("Done")
