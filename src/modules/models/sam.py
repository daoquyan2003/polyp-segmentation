import torch
from segment_anything import sam_model_registry


class SAM:
    def __init__(self, model_type: str, sam_checkpoint: str):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        if torch.cuda.is_available:
            self.sam.to("cuda")
        else:
            self.sam.to("cpu")
