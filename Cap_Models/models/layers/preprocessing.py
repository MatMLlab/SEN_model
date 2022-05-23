"""
Preprocessing codes
"""


import numpy as np
from monty.json import MSONable



class Scaler(MSONable):


    def transform(self, target, n = 1):

        raise NotImplementedError

    def inverse_transform(self, transformed_target, n = 1):

        raise NotImplementedError


class StandardScaler(Scaler):


    def __init__(self, mean = 0.0, std = 1.0, is_intensive = True):

        self.mean = mean
        if np.abs(std) < np.finfo(float).eps:
            std = 1.0
        self.std = std
        self.is_intensive = is_intensive

    def transform(self, target, n = 1):

        if self.is_intensive:
            n = 1
        return (target / n - self.mean) / self.std

    def inverse_transform(self, transformed_target, n = 1):

        if self.is_intensive:
            n = 1
        return n * (transformed_target * self.std + self.mean)

    @classmethod
    def from_training_data(
        cls, structures, targets, is_intensive = True):

        if is_intensive:
            new_targets = targets
        else:
            new_targets = [i / len(j) for i, j in zip(targets, structures)]
        mean = np.mean(new_targets).item()
        std = np.std(new_targets).item()
        return cls(mean, std, is_intensive)

    def __str__(self):
        return "StandardScaler(mean=%.3f, std=%.3f, is_intensive=%d)" % (self.mean, self.std, self.is_intensive)

    def __repr__(self):
        return str(self)


class DummyScaler(MSONable):

    @staticmethod
    def transform(target, n = 1):

        return target

    @staticmethod
    def inverse_transform(transformed_target, n = 1):

        return transformed_target

    @classmethod
    def from_training_data(cls, structures, targets, is_intensive = True):

        return cls()
