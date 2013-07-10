from tempfile import mkstemp
import os
from itertools import islice

import numpy as np
import pytest

from ..model import Model
from ..extractors import RGBExtractor
from ..dr1 import LocationGenerator


class TestModel(object):

    def setup_method(self, method):
        m = Model(extractor=RGBExtractor(),
                  locator=LocationGenerator())
        on = m.locator.positives()[:5]
        off = list(islice(m.locator.negatives_iterator(), 0, 5))

        self.m = m
        self.on = on
        self.off = off

    def test_first_fit(self):

        self.m.first_fit(self.on, self.off)
        assert hasattr(self.m.estimator, 'estimators_')

    def test_add_layer(self):

        self.m.first_fit(self.on, self.off)
        n = len(self.m.estimator.estimators_)
        self.m.add_layer(self.on, self.off)
        assert len(self.m.estimator.estimators_) == n + 1

    def test_false_positives(self):

        fp = self.m.false_positives(5)
        assert len(fp) == 5
        assert self.m.predict(fp).all()

        self.m.first_fit(self.on, fp)

        assert not self.m.predict(fp).all()

        fp = self.m.false_positives(5)
        assert self.m.predict(fp).all()

    def test_retained_data(self):

        self.m.first_fit(self.on, self.off)
        self.m.add_layer(self.on[:1], self.off[:1])
        self.m.add_layer(self.on[1:], self.off[1:])

        td = self.m.training_data

        assert len(td) == 3
        assert td[0]['pos'] == self.on
        assert td[0]['neg'] == self.off
        assert td[1]['pos'] == self.on[:1]
        assert td[1]['neg'] == self.off[:1]
        assert td[2]['pos'] == self.on[1:]
        assert td[2]['neg'] == self.off[1:]

    def _assert_copies(self, m1, m2):
        assert m1.training_data == m2.training_data
        np.testing.assert_array_equal(m1.predict(self.on),
                                      m2.predict(self.on))

    def test_io(self):

        self.m.first_fit(self.on, self.off)
        self.m.add_layer(self.on, self.off)

        try:
            file, path = mkstemp('')
            self.m.save(path)
            m2 = self.m.load(path)
        finally:
            assert os.stat(path).st_size < 1024 ** 3
            os.unlink(path)

        self._assert_copies(self.m, m2)

    @pytest.mark.skipif('True')
    def test_cloud_false_pos(self):

        o = self.m.cloud_false_positives(100, workers=10)
        assert len(o) == 100

        for i in range(len(o)):
            assert self.m.predict(o[i])
            for j in range(i + 1, len(o)):
                assert o[i] != o[j]
