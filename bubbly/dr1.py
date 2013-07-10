"""
Code in this file deals with the data from the
Milky Way Project First Data Release
"""
import os
import cPickle as pickle

import numpy as np
import h5py
from cloud import bucket, running_on_cloud

from .util import overlap


def _has_field(lon):
    """Return True if image data exists for a given longitude"""
    base = os.path.join(os.path.dirname(__file__), 'data', 'galaxy',
                        'registered')
    if not running_on_cloud():
        return os.path.exists(os.path.join(base, "%3.3i_i4.fits" % lon)) and \
            os.path.exists(os.path.join(base, "%3.3i_mips.fits" % lon))
    else:
        return bucket.exists("%3.3i_i4.fits" % lon) and \
            bucket.exists("%3.3i_mips.fits" % lon)


def _on_args(catalog, row):
    """Given the row of the DR1 catalog, turn into
    (lon, lat, radius) tuple
    """
    lon, lat, pa, a, b, t, hr = catalog[row]
    pa *= -1
    return lon, lat, 1.3 * (a + t)


def _high_quality_on_locations():
    pth = os.path.join(os.path.dirname(__file__), 'data', 'hiq.pkl')
    if os.path.exists(pth):
        return pickle.load(open(pth))

    infile = os.path.join(os.path.dirname(__file__), 'data', 'stamps2.h5')
    cat = get_catalog()
    on_id = np.array(map(int, h5py.File(infile, 'r')['on'].keys()))

    rgbs = h5py.File(infile, 'r')['on'].values()

    #has green channel...
    def has_green(rgb):
        return (rgb[:, :, 1] == 0).mean() < 0.3

    good = np.array([has_green(r) for r in rgbs], dtype=np.bool)

    #...and hit rate > 0.3
    good = good & (cat[on_id, -1] > 0.3)
    result = [_on_args(cat, row) for row in on_id[good]]

    with open(pth, 'w') as outfile:
        pickle.dump(result, outfile)

    return result


def get_catalog():
    pth = os.path.join(os.path.dirname(__file__), 'data', 'catalog.pkl')

    if running_on_cloud():
        return pickle.load(bucket.getf('catalog.pkl'))

    return pickle.load(open(pth))


def on_stamp_params():
    """Iterate over high quality + examples,
    yield tuples of (field_lon, lcen, bcen, r)

    These can be fed to Extractor objects
    """
    for l, b, r in _high_quality_on_locations():
        yield int(np.round(l)), l, b, r


class LocationGenerator(object):

    def __init__(self, mod3=0):
        if int(mod3) != mod3 or int(mod3) not in [0, 1, 2]:
            raise ValueError("mod3 must be one of (0, 1, 2)")
        self.mod3 = int(mod3)

    def __eq__(self, other):
        if not isinstance(other, LocationGenerator):
            return False
        return self.mod3 == other.mod3

    def positives(self):
        return [p for p in on_stamp_params() if (p[0] % 3) == self.mod3]

    def _random_field(self):
        while True:
            l = np.random.randint(0, 361, 1)[0]
            if ((l % 3) == self.mod3) and _has_field(l):
                return  l

    def negatives_iterator(self):
        """Yield an infinite sequence of offset stamp parameters

        Yields
        ------
        An infinite sequence of (field_lon, lcen, bcen, rad)
        """
        cat = get_catalog()
        l0, b0, r0 = cat[:, 0], cat[:, 1], cat[:, 3]

        while True:
            lon = self._random_field()
            nsample = 30000 if running_on_cloud() else 500

            l = np.random.uniform(-.5, .5, nsample) + lon
            b = np.random.uniform(-.8, .8, nsample)
            r = np.random.choice(r0, size=nsample)

            bad = overlap(l, b, r, l0, b0, r0)
            l, b, r = l[~bad], b[~bad], r[~bad]

            for i in range(l.size):
                yield lon, l[i], b[i], r[i]
