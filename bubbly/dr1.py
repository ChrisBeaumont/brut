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
    pth = os.path.join(os.path.dirname(__file__), 'data', 'vetted.txt')
    rows = map(int, open(pth).readlines())
    cat = get_catalog()

    return [_on_args(cat, row) for row in rows]


def get_catalog():
    pth = os.path.join(os.path.dirname(__file__), 'data', 'catalog.pkl')

    if running_on_cloud():
        return pickle.load(bucket.getf('catalog.pkl'))

    return pickle.load(open(pth))

def bubble_params(bubbles):
    c = get_catalog()
    result = []
    for b in bubbles:
        args = _on_args(c, b)
        result.append([int(np.round(args[0])), args[0], args[1], args[2]])
    return result

def on_stamp_params():
    """Iterate over medium quality + examples,
    yield tuples of (field_lon, lcen, bcen, r)

    These can be fed to Extractor objects
    """
    for l, b, r in _high_quality_on_locations():
        yield int(np.round(l)), l, b, r

def highest_quality_on_params():
    """Iterate over highest-confidence + examples
    yield tuples of (field_lon, lcen, bcen, r)
    """
    pth = os.path.join(os.path.dirname(__file__), 'data', 'best', 'data.pkl')
    data = pickle.load(open(pth))
    for k, (l, b, r) in data.items():
        if os.path.exists(os.path.join(os.path.dirname(pth), k)):
            yield int(np.round(l)), l, b, r


class LocationGenerator(object):
    positive_generator = staticmethod(on_stamp_params)

    def __init__(self, mod3=0):
        if int(mod3) != mod3 or int(mod3) not in [0, 1, 2]:
            raise ValueError("mod3 must be one of (0, 1, 2)")
        self.mod3 = int(mod3)

    def __eq__(self, other):
        if not isinstance(other, LocationGenerator):
            return False
        return self.mod3 == other.mod3

    def valid_longitude(self, l):
        return l % 3 == self.mod3

    def positives(self):
        return sorted([p for p in self.positive_generator()
                      if self.valid_longitude(p[0])])

    def cv_positives(self):
        return sorted([p for p in self.positive_generator()
                      if not self.valid_longitude(p[0])])

    def _random_field(self):
        while True:
            l = np.random.randint(0, 361, 1)[0]
            if self.valid_longitude(l) and _has_field(l):
                return  l

    def negatives_iterator(self, samples_per_field=None):
        """Yield an infinite sequence of offset stamp parameters

        Parameters
        ----------
        samples_per_field : int (optional)
            How many samples to generate before switching
            longitues by >1 deg. Tuning this can improve
            IO performance.

        Yields
        ------
        An infinite sequence of (field_lon, lcen, bcen, rad)
        """
        cat = get_catalog()
        l0, b0, r0 = cat[:, 0], cat[:, 1], cat[:, 3]

        while True:
            lon = self._random_field()
            nsample = samples_per_field
            if nsample is None:
                nsample = 30000 if running_on_cloud() else 500

            l = np.random.uniform(-.5, .5, nsample) + lon
            b = np.random.uniform(-.8, .8, nsample)
            r = np.random.choice(r0, size=nsample)

            bad = overlap(l, b, r, l0, b0, r0)
            l, b, r = l[~bad], b[~bad], r[~bad]

            for i in range(l.size):
                yield lon, l[i], b[i], r[i]

class WideLocationGenerator(LocationGenerator):
    def valid_longitude(self, l):
        return l % 3 != self.mod3
