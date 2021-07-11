
import copy
import numpy as np


class PredictionVariation:

    def __init__(self, description, evaluatefn):
        self.description = description
        self.evaluatefn = evaluatefn
        self.accumulated_results = []

    def evaluate(self, predicts, certainties, track):
        self.accumulated_results.append(np.argmax(self.evaluatefn(predicts, certainties, track)))

    def reset(self):
        self.accumulated_results = []


def _sum_weighted(predicts, weights):
    return np.matmul(weights.T, predicts)


_GENERAL_PREDICTIONS = [
    PredictionVariation('Mean prediction', lambda p, c, t: p.sum(axis=0)),
    PredictionVariation('Mean squared prediction', lambda p, c, t: (p**2).sum(axis=0)),
    PredictionVariation('Pixels weighted mean squared prediction', lambda p, c, t: _sum_weighted(p**2, np.array(t.pixels))),
    PredictionVariation('Mass weighted mean squared prediction', lambda p, c, t: _sum_weighted(p**2, np.array(t.masses)))
]

_CERTAINTY_PREDICTIONS = [
    PredictionVariation('Certainty weighted mean prediction', lambda p, c, t: _sum_weighted(p, c)),
    PredictionVariation('Certainty weighted mean squared prediction', lambda p, c, t: _sum_weighted(p**2, c)),
    PredictionVariation('Certainty * pixels weighted mean squared prediction', lambda p, c, t: _sum_weighted(p ** 2, c.flatten()*np.array(t.pixels)))
]

def get_general_predictions():
    return copy.deepcopy(_GENERAL_PREDICTIONS)

def get_predictions_with_certainty():
    return copy.deepcopy(_GENERAL_PREDICTIONS + _CERTAINTY_PREDICTIONS)
