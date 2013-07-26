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

def _merge_rows(stamps, scores, i, j):
    if scores[i] > scores[j]:
        return np.delete(stamps, j, 0)
    else:
        return np.delete(stamps, i, 0)

def nn_similar_size(stamps, size_similarity=2.0):
    """
    Given a stamp array, return the index of the nearest neighbor
    to each point, only considering neighbors of similar size

    Parameters
    ----------
    stamps : ndarray
        stamps array as described in `merge`

    size_similarity : float, optional
        Relative size similarity threshold. Neighbors differ in size
        by less than this fraction

    Returns
    -------
    integer array, giving the neighbor index of each row in stamps.
    Rows with no neighbor are assigned -1
    """
    xy = stamps[:, 1:3]
    r = stamps[:, 3]
    tree = NearestNeighbors()
    tree.fit(xy)

    result = np.zeros(stamps.shape[0], dtype=np.int) - 1

    for i in range(stamps.shape[0]):
        k = 5
        while True:
            ds, ids = tree.kneighbors(xy[i], k)

            ds = ds.ravel()
            ids = ids.ravel()

            other = ids != i
            ds = ds[other]
            ids = ids[other]

            if ds.size == 0:
                break

            ds /= np.sqrt(r[i] * r[ids])
            dmax = ds.max()

            size_ratio = r[i] / r[ids]
            size_ratio = np.maximum(size_ratio, 1 / size_ratio)
            good = size_ratio <= size_similarity
            good &= (ds < 1)

            ds = ds[good]
            ids = ids[good]

            if ds.size > 0:
                result[i] = ids[0]
                break

            if dmax >= 1:
                break

            if k >= stamps.shape[0]:
                break

            k = min(k * 2, stamps.shape[0])

    return result


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
    A row-subset of stamps, containing the unique objects
    """

    stamps = np.atleast_2d(stamps)
    while True:
        if stamps.shape[0] == 1:
            return stamps

        nearest = nn_similar_size(stamps)
        if nearest.max() == -1:
            return stamps
        d = distance(stamps, nearest)
        best = np.argmin(d)
        i, j = best, nearest[best]

        stamps = _merge_rows(stamps, scores, i, j)


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
    """

    a = np.asarray(a)
    b = np.asarray(b)
    #distance matrix
    d = np.hypot(a[:, 1].reshape(-1, 1) - b[:, 1].reshape(1, -1),
                 a[:, 2].reshape(-1, 1) - b[:, 2].reshape(1, -1))

    ra = a[:, -1].reshape(-1, 1)
    rb = b[:, -1].reshape(1, -1)
    bad = (d > np.minimum(ra, rb))
    d[bad] = np.inf

    ratio = np.maximum(ra / rb, rb / ra)
    bad = ratio > 2
    d[bad] = np.inf

    result = np.zeros(a.shape[0], dtype=np.int) - 1

    while np.isfinite(d).any():
        best = np.argmin(d)
        i, j = np.unravel_index(best, d.shape)

        result[i] = j
        d[i, :] = np.inf
        d[:, j] = np.inf

    return result
