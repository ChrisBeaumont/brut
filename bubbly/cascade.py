from math import ceil
from warnings import warn
from functools import wraps

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import gradient_boosting as gb
from sklearn.base import clone


def _recall_bias(df, frac):
    """Choose a bias for a decision function, such that
    at least `frac` fraction of positive examples satisfy

    clf.decision_function(X) >= bias

    Parameters
    ----------
    df: array-like
        Decision function evalued for positive examples
    frac : float
        Minimum fraction of positive examples to pass through

    Returns
    -------
    float
    """
    df.sort()

    tiny = 1e-7
    if frac <= 0:
        return df[-1] + tiny
    if frac >= 1:
        return df[0]

    ind = int((1 - frac) * df.size)
    return df[ind]


def _fpos_bias(df, fpos, tneg):
    """Choose a bias for a decision function, such that
    at most `frac` fraction of negative examples satisfy

    clf.decision_function(X) >= bias

    Parameters
    ----------
    df: array-like
        Decision function evalued for negative examples
    fpos : float
        Maximum false positive fraction
    tneg : int
        Total number of negative examples (may be larger than df,
        if previous negatives have been removed in previous versions
        of the cascade)

    Returns
    -------
    float
    """
    df.sort()

    tiny = 1e-7
    fpos = fpos * tneg / df.size

    if fpos >= 1:
        return df[0] - tiny

    ind = max(min(int((1 - fpos) * df.size), df.size - 1), 0)

    result = df[ind] + tiny
    assert (df >= result).mean() <= fpos

    return result


def _set_bias(clf, X, Y, recall, fpos, tneg):
    """Choose a bias for a classifier such that the classification
    rule

    clf.decision_function(X) - bias >= 0

    has a recall of at least `recall`, and (if possible) a false positive rate
    of at most `fpos`

    Paramters
    ---------
    clf : Classifier
        classifier to use
    X : array-like [M-examples x N-dimension]
        feature vectors
    Y : array [M-exmaples]
        Binary classification
    recall : float
        Minimum fractional recall
    fpos : float
        Desired Maximum fractional false positive rate
    tneg : int
        Total number of negative examples (including previously-filtered
        examples)
    """
    df = clf.decision_function(X).ravel()
    r = _recall_bias(df[Y == 1], recall)
    f = _fpos_bias(df[Y == 1], fpos, tneg)
    return min(r, f)


def needs_fit(func):

    @wraps(func)
    def result(self, *args, **kwargs):
        if not hasattr(self, 'estimators_'):
            raise ValueError("Estimator not fitted, call `fit` "
                             "before making predictions")
        return func(self, *args, **kwargs)

    return result


class CascadedBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_recall=0.99, false_pos=1e-3,
                 max_layers=5, verbose=0,
                 base_clf=None):
        """ A cascaded boosting classifier

        Based on the Viola and Jones (2001) Cascade

        A cascaded classifier aggregates multiple classifiers in
        sequence. Objects are classified as +1 only if every
        individual classifier classifies them as +1.

        Parameters
        ----------
        layer_recall : float
            The minimum recall of training data at each layer
            of the cascade
        false_pos : float
            The desired false positive rate of the entire cascade
        mas_layers : int
            The maximum number of cascade layers to build during `fit`
        verbose : int
            Set verbose > 0 to print informational messages
        base_clf : Classifier instance (optional)
            The classifier object to clone and train at each layer of the
            cascade. Defaults to sklearn.ensemble.GradientBoostingClassifier()
        """

        if base_clf is None:
            base_clf = GradientBoostingClassifier()

        self.base_clf = base_clf

        self.layer_recall = layer_recall
        self.false_pos = false_pos
        self.max_layers = max_layers

        self.converged = False

        self.verbose = verbose

    @needs_fit
    def staged_predict(self, X):
        """Predict classification of `X` for each iteration"""
        result = np.ones(X.shape[0], dtype=np.int)
        for b, c in zip(self.bias, self.estimators_):
            result &= (c.decision_function(X).ravel() >= b)
            yield result.copy()

    @needs_fit
    def staged_decision_function(self, X):
        """Compute decision function of `X` for each iteration"""
        result = np.zeros(X.shape[0])
        for c, b in zip(self.estimators_, self.bias):
            good = result >= 0
            result[good] = c.decision_function(X[good]) - b
            yield result.copy()

    @needs_fit
    def decision_function(self, X):
        """Compute the decision function of `X`"""
        result = np.zeros(X.shape[0])
        for c, b in zip(self.estimators_, self.bias):
            good = result >= 0
            result[good] = c.decision_function(X[good]) - b
        return result

    def fit(self, X, Y):
        """ Fit the casecaded boosting model """

        X, Y = X.copy(), Y.copy()
        nex = Y.size

        tpos = Y.sum()
        tneg = nex - tpos

        F = [1.0]
        self.bias = []
        self.estimators_ = []

        for i in range(self.max_layers):
            #assert self._check_invariants(X, Y, tpos, tneg, i)

            if F[-1] < self.false_pos:
                self.converged = True
                break

            if np.unique(Y).size == 1:  # down to a single class
                self.converged = True
                break

            F.append(F[-1])
            clf = clone(self.base_clf)
            clf.fit(X, Y)

            bias = _set_bias(clf, X, Y, self.layer_recall,
                             self.false_pos, tneg)

            self.estimators_.append(clf)
            self.bias.append(bias)

            Yp = clf.decision_function(X).ravel() >= bias
            F[-1] = 1.0 * ((Y == 0) & Yp).sum() / tneg
            rc = 1.0 * ((Y == 1) & Yp).sum() / tpos
            if self.verbose > 0:
                print ("Cascade round %i. False pos rate: %e. "
                       "Recall: %e" % (i + 1, F[-1], rc))

            if Yp.all():
                warn("Could not filter any more examples. "
                     "False positive rate: %e. Recall: %e" % (F[-1], rc))
                self.converged = False
                break

            X = X[Yp]
            Y = Y[Yp]

        else:
            warn("Could not reduce false positive enough after "
                 "%i layers. False positive rate: %e. Recall: %e" %
                 (self.max_layers, F[-1], rc))
            self.converged = False

        return self

    def add_cascade_layer(self, X, Y):
        """Add another layer to the cascade.

        The new layer will achieve a recall of at least `layer_recall`
        on the input data. It will not apply any false positive criteria
        """
        clf = clone(self.base_clf)
        clf.fit(X, Y)
        bias = _recall_bias(clf.decision_function(X[Y == 1]).ravel(),
                            self.layer_recall)
        self.estimators_.append(clf)
        self.bias.append(bias)

    @needs_fit
    def predict(self, X):
        """Predict class for X"""
        return self.decision_function(X).ravel() >= 0
