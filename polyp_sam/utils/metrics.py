import torch
from monai.metrics import DiceMetric, GeneralizedDiceScore


class Metrics:
    def __init__(self):
        self.dice_metric = DiceMetric()
        self.gd_metric = GeneralizedDiceScore()

    def update(self, outputs, targets):
        self.dice_metric.update(outputs, targets)
        self.gd_metric.update(outputs, targets)

    def compute(self):
        dice_score = self.dice_metric.compute()
        gd_score = self.gd_metric.compute()
        return dice_score, gd_score
