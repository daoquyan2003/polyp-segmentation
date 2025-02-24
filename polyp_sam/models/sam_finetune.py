from segment_anything import sam_model_registry


class SAMFinetune:
    def __init__(self, model_type, checkpoint_path, device):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=device)

    def get_model(self):
        return self.model
