from itertools import islice, ifilter
import logging
import cPickle as pickle

logging.getLogger().setLevel(logging.DEBUG)

import numpy as np
from sklearn.base import clone
import cloud

from .cascade import CascadedBooster


class Model(object):

    def __init__(self, extractor, locator, **kwargs):
        self.estimator = CascadedBooster(**kwargs)
        self.extractor = extractor
        self.locator = locator
        self.training_data = []

    def _default_on_off(self, on=None, off=None):
        if on is None:
            on = self.locator.positives()
        if off is None:
            off = self.false_positives(5 * len(on))
        return on, off

    def _make_xy(self, on, off):

        x = np.vstack(self.extractor(*o).reshape(1, -1) for o in on + off)
        y = np.hstack((np.ones(len(on), dtype=np.int),
                       np.zeros(len(off), dtype=np.int)))

        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2
        assert y.ndim == 1

        return x, y

    def save(self, path):
        with open(path, 'w') as outfile:
            pickle.dump(self, outfile)

    @classmethod
    def load(cls, path):
        with open(path) as infile:
            return pickle.load(infile)

    def predict(self, params):
        if not hasattr(params[0], '__len__'):
            params = [params]

        if not hasattr(self.estimator, 'estimators_'):
            return np.ones(len(params), dtype=np.int)

        x, y = self._make_xy(params, [])
        return self.estimator.predict(x)

    def false_positives(self, num):
        return list(islice(self.false_positives_iterator(), 0, num))

    def false_positives_iterator(self):
        def pos(x):
            return self.predict(x)[0] == 1

        return ifilter(pos, self.locator.negatives_iterator())

    def cloud_false_positives(self, num, workers=10):
        jobs = cloud.map(self.false_positives, [num / workers] * workers,
                         _env='mwp', _type='c1')
        return [r for j in cloud.result(jobs) for r in j]

    def _reset(self):
        self.training_data = []
        self.estimator = clone(self.estimator)

    def add_layer(self, on=None, off=None):
        on, off = self._default_on_off(on, off)
        self.training_data.append(dict(pos=on, neg=off))

        x, y = self._make_xy(on, off)
        self.estimator.add_cascade_layer(x, y)

    def first_fit(self, on=None, off=None):
        on, off = self._default_on_off(on, off)
        self._reset()
        self.training_data = [dict(pos=on, neg=off)]

        x, y = self._make_xy(on, off)
        self.estimator.fit(x, y)
