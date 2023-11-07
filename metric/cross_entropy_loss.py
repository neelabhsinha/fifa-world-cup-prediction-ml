import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    @staticmethod
    def calculate(predictions, targets):
        assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"

        # Avoid numerical instability by adding a small epsilon value to predictions
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Calculate the cross-entropy loss
        loss = -np.sum(targets * np.log(predictions)) / len(predictions)

        return loss
