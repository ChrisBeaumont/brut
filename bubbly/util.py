import os
from cPickle import load, dump

from skimage.transform import resize
from sklearn.metrics import recall_score
import numpy as np


def lon_offset(x, y):
    """Return angular separation between two offsets which possibly
    straddle l=0

    >>> lon_offset(0, 1)
    1
    >>> lon_offset(1, 0)
    1
    >>> lon_offset(0, 355)
    5
    >>> lon_offset(355, 0)
    5
    >>> lon_offset(181, 0)
    179
    """
    return min(abs(x - y), abs(x + 360 - y), abs(x - (y + 360)))


def up_to_date(inputs, output):
    """Test whether an output file is more recent than
    a list of input files

    Parameters
    ----------
    inputs: List of strings (paths to input files)
    output: string (path to output file)

    Returns
    -------
    Boolean (True if output more recent than all inputs)
    """
    if not os.path.exists(output):
        return False

    itime = max(os.path.getmtime(input) for input in inputs)
    otime = os.path.getmtime(output)
    return otime > itime


def scale(x, mask=None, limits=None):
    """Scale an array as is done in MWP paper

    Sqrt transform of data cipped at 5 and 99.8%

    """
    limits = limits or [5, 99.8]
    if mask is None:
        lo, hi = np.percentile(x, limits)
    else:
        lo, hi = np.percentile(x[mask], limits)

    x = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return (np.sqrt(x) * 255).astype(np.uint8)


def resample(arr, shape):
    """Resample a 2D array, to change its shape"""
    # skimage's resize needs scaled data
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    arr = (arr - lo) / (hi - lo)
    result = resize(arr, shape, mode='nearest')
    return result * (hi - lo) + lo


def save_learner(clf, filename):
    """Save a scikit-learn model to a file"""
    with open(filename, 'w') as outfile:
        dump(clf, outfile)


def load_learner(filename):
    """ Load a scikit-learn model from a file"""
    with open(filename) as infile:
        result = load(infile)
    return result


def false_pos(Y, Yp):
    return 1.0 * ((Y == 0) & (Yp == 1)).sum() / (Y == 0).sum()


def recall(Y, Yp):
    return recall_score(Y, Yp)


def summary(clf, x, y):
    df = clf.decision_function(x).ravel()
    yp = df > 0

    print 'False Positive: %0.3f' % false_pos(y, yp)
    print 'Recall:         %0.3f' % recall(y, yp)
    print 'Accuracy:       %0.3f' % (yp == y).mean()


def rfp_curve(yp, Y, **kwargs):
    """ Plot the false positive rate as a function of recall """
    import matplotlib.pyplot as plt

    npos = Y.sum()
    nneg = Y.size - npos
    ind = np.argsort(yp)[::-1]
    y = Y[ind]
    yp = yp[ind]

    recall = (1. * np.cumsum(y == 1)) / npos
    false_pos = (1. * np.cumsum(y == 0)) / nneg

    r = 1.0 * ((yp > 0) & (y == 1)).sum() / npos
    fp = 1.0 * ((yp > 0) & (y == 0)).sum() / nneg

    l, = plt.plot(recall, false_pos, **kwargs)
    plt.plot([r], [fp], 'o', c=l.get_color())
    plt.xlabel('Recall')
    plt.ylabel('False Positive')
    plt.title("R=%0.3f, FP=%0.4f" % (r, fp))
    ax = plt.gca()

    ax.grid(which='major', axis='x',
            linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x',
            linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y',
            linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y',
            linewidth=0.25, linestyle='-', color='0.75')

    return recall, false_pos


def _stamp_distances(stamps):
    #compute distance matrix for a list of stamps
    n = len(stamps)
    result = np.zeros((n, n)) * np.nan
    for i in range(n):
        si = stamps[i]
        xi, yi, di = si[1:4]
        for j in range(i + 1, n, 1):
            sj = stamps[j]
            xj, yj, dj = sj[1:4]
            dx = np.hypot(xi - xj, yi - yj)
            if dx > max(di, dj):
                continue
            elif max(di / dj, dj / di) > 3:
                continue
            else:
                d = dx / ((di + dj) / 2.)
                result[i, j] = result[j, i] = d
    return result


def _decimate(dist_matrix, scores):
    inds = np.arange(dist_matrix.shape[0])

    while True:
        if ~np.isfinite(dist_matrix).any():
            break
        best = np.nanargmin(dist_matrix)
        i, j = np.unravel_index(best, dist_matrix.shape)
        merge = i if scores[i] < scores[j] else j
        inds = np.delete(inds, merge)
        scores = np.delete(scores, merge)
        dist_matrix = np.delete(np.delete(dist_matrix, merge, 0), merge, 1)
    return inds


def merge_detections(detections):
    locations, scores = zip(*detections)
    scores = np.array(scores)
    dist = _stamp_distances(locations)
    result = _decimate(dist, scores)
    return np.asarray(detections)[result]


def normalize(arr):
    """Flatten and L2-normalize an array, and return"""
    arr = arr.ravel().astype(np.float)
    n = np.sqrt((arr ** 2).sum())
    return arr / n


ely, elx = np.mgrid[:40, :40]


def ellipse(x0, y0, a, b, dr, theta0):
    """Make a 40x40 pix image of an ellipse"""

    r = np.hypot(elx - x0, ely - y0)
    theta = np.arctan2(ely - y0, elx - x0) - np.radians(theta0)
    r0 = a * b / np.hypot(a * np.cos(theta), b * np.sin(theta))
    return np.exp(-np.log(r / r0) ** 2 / (dr / 10.) ** 2)


def _sample_and_scale(i4, mips, do_scale, limits):

    i4 = resample(i4, (40, 40))
    mips = resample(mips, (40, 40))

    assert i4.shape == (40, 40), i4.shape
    assert mips.shape == (40, 40), mips.shape

    mask = mips > 0

    if do_scale:
        try:
            i4 = scale(i4, limits=limits)
            mips = scale(mips, mask, limits=limits)
            mips[~mask] = 255
        except ValueError:
            #print 'Could not rescale images (bad pixels?)'
            return
    else:
        mips[~mask] = np.nan

    rgb = np.dstack((mips, i4, i4 * 0))
    return rgb


def _unpack(tree):
    if isinstance(tree, np.ndarray):
        return tree.ravel()
    return np.hstack(_unpack(t) for t in tree)


def multiwavelet_from_rgb(rgb):
    from scipy.fftpack import dct
    from pywt import wavedec2

    r = rgb[:, :, 0].astype(np.float)
    g = rgb[:, :, 1].astype(np.float)

    dctr = dct(r, norm='ortho').ravel()
    dctg = dct(g, norm='ortho').ravel()
    daubr = _unpack(wavedec2(r, 'db4'))
    daubg = _unpack(wavedec2(g, 'db4'))
    return np.hstack([dctr, dctg, daubr, daubg])
