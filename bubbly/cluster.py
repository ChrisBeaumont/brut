import numpy as np

from sklearn.neighbors import NearestNeighbors


def distance(s, inds):
    """
    Distance between pairs of stamps, normalized by
    stamp size

    """
    x = s[:, 1]
    y = s[:, 2]
    r = s[:, 3]
    result = np.hypot(x - x[inds], y - y[inds])
    result /= np.sqrt(r * r[inds])
    result[inds == -1] = np.inf

    return result

def _merge(result, dist, i, j, scores):
    kill = i if scores[i] < scores[j] else j
    result[kill] = False

    dist[kill, :] = np.inf
    dist[:, kill] = np.inf

def merge(stamps, scores):
    """
    Merge a collection of overlapping stamps with scores

    Parameters
    ----------
    stamps : array-like (n x 4)
        Array of stamp parameters. Each row consists of (lfield, l, b, r)

    scores : array-like (n elements)
        Classification scores

    Returns
    -------
    A tuple of stamps_subset, scores_subset, each containing
    a row subset of the original arrays
    """
    stamps = np.atleast_2d(stamps)

    d = bubble_distance_matrix(stamps, stamps)

    eye = np.arange(d.shape[0])
    d[eye, eye] = np.inf

    result = np.ones(d.shape[0], dtype=np.bool)

    while np.isfinite(d).any():
        best = np.argmin(d)
        i, j = np.unravel_index(best, d.shape)
        _merge(result, d, i, j, scores)

    return stamps[result], scores[result]


def bubble_distance_matrix(a, b):

    d = np.hypot(a[:, 1].reshape(-1, 1) - b[:, 1].reshape(1, -1),
                 a[:, 2].reshape(-1, 1) - b[:, 2].reshape(1, -1))

    ra = a[:, -1].reshape(-1, 1)
    rb = b[:, -1].reshape(1, -1)
    size = np.minimum(ra, rb)

    d /= size

    bad = d >= 1
    d[bad] = np.inf

    ratio = np.maximum(ra / rb, rb / ra)
    bad = ratio > 2
    d[bad] = np.inf
    return d


def xmatch(a, b):
    """Match two lists of bubble locations

    The matches are constrained to be one-to-one or
    0-to-one. There are no many-to-one matches

    Matching happens greedily -- each pair of
    un-matched closest bubbles are matched, until
    no more matches are possible


    Parameters
    ----------
    a : list of stamp params
        Each param is a tuple of the form (lon_field, lon, lat, radius)

    b : list of stamp params
        like a

    These stamp params are returned by, e.g., Field.all_stamps
    or bubbly.dr1.bubble_params()

    Returns
    -------
    match : array of ints
        Each element lists which item in b matches to each item in a,
        or -1 if no match is found
    """

    a = np.asarray(a)
    b = np.asarray(b)
    d = bubble_distance_matrix(a, b)

    result = np.zeros(a.shape[0], dtype=np.int) - 1

    while np.isfinite(d).any():
        best = np.argmin(d)
        i, j = np.unravel_index(best, d.shape)

        result[i] = j
        d[i, :] = np.inf
        d[:, j] = np.inf

    return result
