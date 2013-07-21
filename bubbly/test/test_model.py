from tempfile import mkstemp
import os
from itertools import islice

import numpy as np
import pytest

from ..model import Model, ModelGroup
from ..extractors import RGBExtractor
from ..dr1 import LocationGenerator, WideLocationGenerator
from ..cascade import CascadedBooster


class TestModel(object):

    def setup_method(self, method):
        m = Model(extractor=RGBExtractor(),
                  locator=LocationGenerator(),
                  classifier=CascadedBooster())
        on = m.locator.positives()[:5]
        off = list(islice(m.locator.negatives_iterator(), 0, 5))

        self.m = m
        self.on = on
        self.off = off

    def test_fit(self):

        self.m.fit(self.on, self.off)
        assert hasattr(self.m.classifier, 'estimators_')

    def test_retrain(self):

        self.m.fit(self.on, self.off)
        m2 = Model(extractor=RGBExtractor(),
                  locator=LocationGenerator(),
                  classifier=CascadedBooster())
        m2.retrain(self.m.training_data)
        assert m2.training_data == self.m.training_data

    def test_add_layer(self):

        self.m.fit(self.on, self.off)
        n = len(self.m.classifier.estimators_)
        self.m.add_layer(self.on, self.off)
        assert len(self.m.classifier.estimators_) == n + 1

    def test_false_positives(self):

        fp = self.m.false_positives(5)
        assert len(fp) == 5
        assert self.m.predict(fp).all()

        self.m.fit(self.on, fp)

        assert not self.m.predict(fp).all()

        fp = self.m.false_positives(5)
        assert self.m.predict(fp).all()

    def test_retained_data(self):

        self.m.fit(self.on, self.off)
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

        self.m.fit(self.on, self.off)
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

    def test_invalid_fit_longitude(self):
        with pytest.raises(ValueError) as exc:
            bad_par = (31, 31, 0, .1)
            self.m.fit(on=[bad_par],
                       off=[bad_par])
        assert exc.value.args[0].startswith('Cannot use this data')

class TestModelGroup(object):

    def test_correct_dispatch(self):
        ms = [Model(extractor=RGBExtractor(),
                    locator=WideLocationGenerator(i),
                    classifier=CascadedBooster()) for i in [0, 1, 2]]
        mg = ModelGroup(*ms)
        assert mg._choose_model(0) == ms[0]
        assert mg._choose_model(1) == ms[1]
        assert mg._choose_model(2) == ms[2]
        assert mg._choose_model(3) == ms[0]

        with pytest.raises(ValueError):
            mg._choose_model(4.1)
