import numpy as np

#Don't know why this is here. I am leaving it as is.
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

#I am assuming data is feteched here. Validation,training and test split is also done here.
# Class split 
# k-Fold CV?
# Anything else?
#Send the data to train.py
#Maintain consistency  logistic_regression -> (preprocess-> train -> test (eval) ) .Things in brackets should happen consistently.(Why not one file?)
#  
# 