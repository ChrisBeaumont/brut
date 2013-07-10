import pytest
import numpy as np


from ..dr1 import LocationGenerator


class TestLocationGenerator(object):

    @pytest.mark.parametrize('mod3', [-1, 1.1, 3, 4])
    def test_invalid_mod3(self, mod3):
        with pytest.raises(ValueError) as exc:
            LocationGenerator(mod3)

    def test_positives(self):
        lg = LocationGenerator(mod3=0)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        assert (lon % 3 == 0).all()

        lg = LocationGenerator(mod3=1)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        assert (lon % 3 == 1).all()

        lg = LocationGenerator(mod3=2)
        p = lg.positives()
        lon = np.array([pp[0] for pp in p])
        assert (lon % 3 == 2).all()

    @pytest.mark.parametrize('mod3', [0, 1, 2])
    def test_random_field(self, mod3):
        lc = LocationGenerator(mod3=mod3)
        for i in range(100):
            assert (lc._random_field() % 3) == mod3

    @pytest.mark.parametrize('mod3', [0, 1, 2])
    def test_off_fields(self, mod3):
        from itertools import islice
        lc = LocationGenerator(mod3)
        for p in islice(lc.negatives_iterator(), 0, 1000):
            assert (p[0] % 3) == mod3
            assert (int(np.round(p[1])) % 3) == mod3

    def test_pickle(self):
        from cPickle import loads, dumps

        lc = LocationGenerator(1)

        ser = dumps(lc)
        assert len(ser) < 200

        lc2 = loads(ser)

        assert lc2 == lc
