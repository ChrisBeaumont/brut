import pytest
import numpy as np


from ..dr1 import LocationGenerator, WideLocationGenerator


class TestLocationGenerator(object):
    LG = LocationGenerator

    def assert_lon_valid(self, lon, mod):
        assert (np.asarray(lon) % 3 == mod).all()

    @pytest.mark.parametrize('mod3', [-1, 1.1, 3, 4])
    def test_invalid_mod3(self, mod3):
        with pytest.raises(ValueError) as exc:
            self.LG(mod3)

    def test_positives(self):
        lg = self.LG(mod3=0)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        self.assert_lon_valid(lon, 0)

        lg = self.LG(mod3=1)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        self.assert_lon_valid(lon, 1)

        lg = self.LG(mod3=2)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        self.assert_lon_valid(lon, 2)

    @pytest.mark.parametrize('mod3', [0, 1, 2])
    def test_random_field(self, mod3):
        lc = self.LG(mod3=mod3)
        for i in range(100):
            self.assert_lon_valid(lc._random_field(), mod3)

    @pytest.mark.parametrize('mod3', [0, 1, 2])
    def test_off_fields(self, mod3):
        from itertools import islice
        lc = self.LG(mod3)
        for p in islice(lc.negatives_iterator(), 0, 1000):
            self.assert_lon_valid(p[0], mod3)
            self.assert_lon_valid(int(np.round(p[1])), mod3)

    def test_pickle(self):
        from cPickle import loads, dumps

        lc = self.LG(1)

        ser = dumps(lc)
        assert len(ser) < 200

        lc2 = loads(ser)

        assert lc2 == lc

class TestWideLocationGenerator(object):
    LG = WideLocationGenerator
    def assert_lon_valid(self, lon, mod):
        assert (np.asarray(lon) % 3 != mod).all()
