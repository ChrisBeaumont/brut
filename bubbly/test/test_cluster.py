import numpy as np

from ..cluster import merge

def test_merge_separate():
    data = [[1, 1, 0, 1], [5, 5, 0, 1]]
    scores = [1, 2]
    np.testing.assert_array_equal(merge(data, scores)[0], data)

def test_merge_merged():
    data = [[0, 0, 0, 6], [5, 5, 0, 6]]
    scores = [2, 1]
    np.testing.assert_array_equal(merge(data, scores)[0], [data[0]])
    np.testing.assert_array_equal(merge(data[::-1], scores[::-1])[0],
                                  [data[0]])

def test_merge_3way():
    data = [[1, 1, 0, 1], [1, 1.1, 0, 1], [2, 2, 0, 1]]
    scores = [2, 1, 1]
    np.testing.assert_array_equal(merge(data, scores)[0], [data[0], data[2]])

def test_merge_3way_sizes():
    data = [[1, 1, 0, 1], [1, 1.1, 0, 1], [1, 1, 0, 5]]
    scores = [2, 1, 1]
    np.testing.assert_array_equal(merge(data, scores)[0], [data[0], data[2]])
