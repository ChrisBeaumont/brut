"""
A simple interface for random exploration of hyperparameter space

"""
import random

import numpy as np
from scipy import stats
from sklearn.metrics import auc
from sklearn import metrics as met


class Choice(object):
    """Randomly select from a list"""
    def __init__(self, *choices):
        self._choices = choices

    def rvs(self):
        return random.choice(self._choices)


class Space(object):
    """
    Spaces gather and randomly sample
    collections of hyperparameters

    Any class with an rvs method is a valid hyperparameter
    (e.g., anything in scipy.stats is a hyperparameter)
    """
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def __iter__(self):
        while True:
            yield {k: v.rvs() for k, v in self._hyperparams.items()}


def auc_below_fpos(y, yp, fpos):
    """
    Variant on the area under the ROC curve score

    Only integrate the portion of the curve
    to the left of a threshold in fpos
    """
    fp, tp, th = met.roc_curve(y, yp)
    good = (fp <= fpos)
    return auc(fp[good], tp[good])


def fmin(objective, space, threshold=np.inf):
    """
    Generator that randomly samples a space,
    and yields whenever a new minimum is encountered

    Parameters
    ----------
    objective : A function which takes hyperparameters
                as input, and computes an objective function and classifier
                out output

    space : the Space to sample

    threshold : A threshold in the objective function values.
                If provided, will not yield anything until
                the objective function falls below threshold

    Yields
    ------
    Tuples of (objective function, parameter dict, classifier)
    """
    best = threshold

    try:
        for p in space:
            f, clf = objective(**p)
            if f < best:
                best = f
                yield best, p, clf
    except KeyboardInterrupt:
        pass


#default space for Gradient Boosted Decision trees
gb_space = Space(learning_rate = stats.uniform(1e-3, 1 - 1.01e-3),
                 n_estimators = Choice(50, 100, 200),
                 max_depth = Choice(1, 2, 3),
                 subsample = stats.uniform(1e-3, 1 - 1.01e-3))


#default space for WiseRF random forests
rf_space = Space(n_estimators = Choice(200, 400, 800, 1600),
                 min_samples_split = Choice(1, 2, 4),
                 criterion = Choice('gini', 'gainratio', 'infogain'),
                 max_features = Choice('auto'),
                 n_jobs = Choice(2))
