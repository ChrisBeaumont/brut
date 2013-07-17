import pytest
import numpy as np

from ..util import *


def test_lon_offset():

    assert lon_offset(0, 1) == 1
    assert lon_offset(1, 0) == 1
    assert lon_offset(0, 355) == 5
    assert lon_offset(355, 0) == 5
    assert lon_offset(181, 0) == 179


def monotonic(x, y):
    ind = np.argsort(x)
    return (np.sort(y) == y[ind]).all()


def check_scale(x, y):

    assert y.dtype == np.uint8
    assert y.max() == 255
    assert y.min() == 0
    assert monotonic(x, y)


def test_scale():

    x = np.random.normal((0, 1), (1000))

    y = scale(x)
    check_scale(x, y)

    y2 = scale(x, limits=[4, 99.8])
    check_scale(x, y2)
    assert (y2 >= y).all()

    y2 = scale(x, limits=[6, 99.9])
    check_scale(x, y2)
    assert (y2 <= y).all()

    y2 = scale(x, limits=[5, 90])
    check_scale(x, y2)
    assert (y2 >= y).all()

    y2 = scale(x, limits=[5, 100])
    check_scale(x, y2)
    assert (y2 <= y).all()

    x2 = np.hstack((x, [5000]))
    mask = np.ones(x2.shape, dtype=np.bool)
    mask[-1] = 0
    y2 = scale(x2, mask=mask)
    np.testing.assert_array_equal(y, y2[:-1])


def test_resample():

    x, y = np.mgrid[-10:10, -10:10]
    r = np.hypot(x, y) / 10
    x = np.sin(r)

    for shp in [(40, 40), (10, 10), (10, 20), (30, 20),
                (20, 30), (20, 10)]:
        y = resample(x, shp)
        assert y.shape == shp
        assert y.dtype == x.dtype

    y = resample(x, (40, 40))
    z = resample(y, (20, 20))
    assert np.abs((z - x)).max() < .1


def test_false_pos():

    y = np.array([True, False, True, False])

    yp = np.array([True, True, True, True])
    assert false_pos(y, yp) == 1

    yp = np.array([True, False, False, True])
    assert false_pos(y, yp) == .5

    yp = np.array([False, False, False, False])
    assert false_pos(y, yp) == 0


def test_chunk():

    x = range(5)

    assert chunk(x, 1) == [x]
    assert chunk(x, 2) == [[0, 1, 2], [3, 4]]
    assert chunk(x, 3) == [[0, 1], [2, 3], [4]]
    assert chunk(x, 4) == [[0, 1], [2, 3], [4]]
    assert chunk(x, 5) == [[0], [1], [2], [3], [4]]

    with pytest.raises(ValueError):
        chunk(x, 6)

    with pytest.raises(ValueError):
        chunk(x, 0)
