from itertools import islice, ifilter
import warnings
import logging
import cPickle as pickle
import json

logging.getLogger(__name__).setLevel(logging.DEBUG)

import numpy as np
from sklearn.base import clone


from .decorators import profile
from .util import chunk, cloud_map


class Model(object):

    def __init__(self, extractor, locator, classifier):
        """ Bundles together an extractor, locator, and classifier """

        self.classifier = classifier
        self.extractor = extractor
        self.locator = locator
        self.training_data = []

    def _default_on_off(self, on=None, off=None):
        if on is None:
            on = self.locator.positives()
        if off is None:
            off = self.false_positives(3 * len(on))
        return on, off

    @profile
    def make_xy(self, on, off):
        x = np.vstack(self.extractor(*o).reshape(1, -1) for o in on + off)
        y = np.hstack((np.ones(len(on), dtype=np.int),
                       np.zeros(len(off), dtype=np.int)))

        #sklearn doesn't like non-finite values, which
        #occasionally popup
        if not np.isfinite(x).all():
            warnings.warn("Non-finite values in feature vectors. Fixing")
            x = np.nan_to_num(x)

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

    def save_training_data(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.training_data, outfile)

    @staticmethod
    def load_training_data(path):
        with open(path) as infile:
            return json.load(infile)

    def predict(self, params):
        if not hasattr(params[0], '__len__'):
            params = [params]

        x, y = self.make_xy(params, [])
        try:
            return self.classifier.predict(x)
        except ValueError:  # not yet fit
            #having an empty model predict 1
            #makes it convenient to generate
            #initial false positives
            return np.ones(len(params), dtype=np.int)

    @profile
    def false_positives(self, num):
        logging.getLogger(__name__).debug("Locally scanning for %i "
                                          "false positives" % num)
        return list(islice(self.false_positives_iterator(), 0, num))

    def false_positives_iterator(self):
        def pos(x):
            return self.predict(x)[0] == 1

        return ifilter(pos, self.locator.negatives_iterator())

    @profile
    def cloud_false_positives(self, num=0, workers=10, jobs=None):
        """
        Use PiCloud to find false positives

        Usage:

        cloud_false_positives(15)

        or

        cloud_false_positives(jobs=range(15, 20))

        Parameters
        ----------
        num : int
            number of false positives to search for
        workers : int (optional, default=10)
            number of PiCloud jobs to use
        jobs : sequence of ints (optional)
            If provided, re-fetch a previous patch
            a previous batch of PiCloud jobs
        """
        results = cloud_map(self.false_positives,
                            [num / workers] * workers,
                            jobs)
        return [r for j in results for r in j]

    def cloud_decision_function(self, x, workers=10, jobs=None):
        results = cloud_map(self.decision_function,
                            chunk(x, workers),
                            jobs)
        return np.hstack(results)

    def decision_function(self, x):
        """
        Compute the decision function for a list of stamp descriptions

        Parameters
        ----------
        x : List of stamp description tuples

        Returns
        -------
        An ndarray of the decision function for each feature extracted
        from x
        """
        result = np.empty(len(x))
        for i, ex in enumerate(x):
            X, _ = self.make_xy([ex], None).reshape(1, -1)
            df = self.classifier.decision_function(X).ravel()
            result[i] = df
        return result

    def _reset(self):
        self.training_data = []
        self.classifier = clone(self.classifier)

    def add_layer(self, on=None, off=None):
        on, off = self._default_on_off(on, off)
        self.training_data.append(dict(pos=on, neg=off))

        x, y = self.make_xy(on, off)
        self.classifier.add_cascade_layer(x, y)

    def retrain(self, training_data):
        self.fit(training_data[0]['pos'], training_data[0]['neg'])
        for td in training_data[1:]:
            self.add_layer(td['pos'], td['neg'])

    @profile
    def fit(self, on=None, off=None):
        self._reset()
        on, off = self._default_on_off(on, off)
        self.training_data = [dict(pos=on, neg=off)]

        x, y = self.make_xy(on, off)
        logging.getLogger(__name__).debug("Fitting")
        self.classifier.fit(x, y)
