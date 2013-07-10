"""
Extractor classes build feature vectors from postage stamp descriptions
"""
import zlib
from itertools import product

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as minimize

from .field import new_field
from .util import normalize, ellipse, multiwavelet_from_rgb


class Extractor(object):
    def __init__(self):
        self._lon = None
        self._field = None

    def setup_field(self, lon):
        """
        Prepare an extractor by loading an appropriate field

        Parameters
        ----------
        lon : int
            The longitude of the field to load

        Notes
        -----
        Fields are cached so, if setup_field is called multiple times in
        a row with the same `lon` value, the data are only loaded once
        """
        if lon == self._lon:
            return
        self._field = None  # de-allocate memory, for cloud machines
        self._lon = lon
        self._field = new_field(lon)

    def __getstate__(self):
        result = self.__dict__.copy()
        result['_field'] = None
        result['_lon'] = None
        return result

    def __call__(self, lon, l, b, r):
        return self.extract(lon, l, b, r)

    def extract(self, lon, l, b, r):
        """
        Extract a feature vector from a postage stamp description

        Paramters
        ---------
        lon : int
            The longitude of the field to use
        l : float
            Bubble center longitude, degrees
        b : float
            Bubble center latitude, degrees
        r : float
            Bubble radius, degrees

        Returns
        -------
        A feature vector. The contents of the feature vector
        are configured by subclasses
        """
        self.setup_field(lon)
        rgb = self._field.extract_stamp(l, b, r, limits=[1, 97])

        if rgb is None:
            rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        elif (rgb[:, :, 1] == 0).mean() > 0.1:
            rgb = np.zeros((40, 40, 3), dtype=np.uint8)

        return self._extract_rgb(rgb)

    def _extract_rgb(self, rgb):
        raise NotImplementedError()


class RGBExtractor(Extractor):
    """Extracts RGB intensity values"""
    def _extract_rgb(self, rgb):
        return rgb


class MultiWaveletExtractor(Extractor):
    """Extracts normalized wavelet coefficients"""
    def _extract_rgb(self, rgb):
        return normalize(multiwavelet_from_rgb(rgb).reshape(1, -1))


class CompressionExtractor(Extractor):
    """Extracts the length of the compressed string of each color"""
    def _extract_rgb(self, rgb):
        clen = lambda x: np.array([len(zlib.compress(x.tostring()))])
        return np.hstack(map(clen, [rgb, rgb[:, :, 0], rgb[:, :, 1],
                                    rgb[:, :, 0] - rgb[:, :, 1]])).reshape(1, -1)


class RawStatsExtractor(Extractor):
    """Extracts percentiles, differences from each channel"""
    def extract(self, lon, l, b, r):
        self.setup_field(lon)
        rgb = self._field.extract_stamp(l, b, r, do_scale=False)
        if rgb is None:
            rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        elif (rgb[:, :, 1] == 0).mean() > 0.1:
            rgb = np.zeros((40, 40, 3), dtype=np.uint8)

        return self._extract_rgb(rgb)

    def _extract_rgb(self, rgb):
        r = rgb[:, :, 0].astype(np.float).ravel()
        g = rgb[:, :, 1].astype(np.float).ravel()
        r[~np.isfinite(r)] = 0
        g[~np.isfinite(g)] = 0
        sums = np.array([r.sum(), g.sum()])
        ps = [1, 3, 5, 10, 50, 90, 95, 97, 99]
        rper = np.array(np.percentile(r, ps))
        gper = np.array(np.percentile(g, ps))
        rdiff = rper[[5, 6, 7, 8]] - rper[[0, 1, 2, 3]]
        gdiff = gper[[5, 6, 7, 8]] - gper[[0, 1, 2, 3]]

        return np.hstack([sums, rper, gper, rdiff, gdiff]).reshape(1, -1)


class EllipseExtractor(Extractor):
    """ Measures overlap with ellipses for each channel"""
    def __init__(self):
        super(EllipseExtractor, self).__init__()

        aas = np.linspace(8, 18, 3)
        bs = aas
        x0s = np.linspace(5, 35, 5)
        y0s = np.linspace(5, 35, 5)
        drs = [4]
        thetas = np.linspace(0, 180, 4)

        params = np.vstack(product(x0s, y0s, aas, bs, drs, thetas))
        templates = np.column_stack([ellipse(x0, y0, a, b, dr, theta).ravel()
                                     for x0, y0, a, b, dr, theta
                                     in params])

        templates /= np.sqrt((templates ** 2).sum(axis=0))
        self.templates = templates
        self.params = params

    def best_match(self, arr):
        arr = normalize(arr)
        dot = np.dot(arr, self.templates)
        best = np.argmax(dot)
        return self.params[best]

    def _fit_ellipse(self, arr):
        narr = normalize(arr)
        bounds = [(10, 30), (10, 30), (3, 30), (3, 30), (.5, 3.5), (0, 360)]

        def fun(x):
            e = ellipse(20, 20, *x[2:]).ravel().astype(np.float)
            norm = np.sqrt((e ** 2).sum())
            e = ellipse(*x).ravel().astype(np.float) / norm
            return np.dot(narr, e) * (-1)

        x0 = self.best_match(narr)
        result = minimize(fun, x0, bounds=bounds, approx_grad=True)
        return result[0], fun(result[0])

    def _extract_rgb(self, rgb):
        r, g = rgb[:, :, 0], rgb[:, :, 1]

        rp, rs = self._fit_ellipse(r)
        gp, gs = self._fit_ellipse(g)
        return np.hstack([rp, rs, gp, gs]).reshape(1, -1)


class RingExtractor(Extractor):
    """ Measures overlap with rings """
    def __init__(self):
        super(RingExtractor, self).__init__()

        y, x = np.mgrid[0:40, 0:40].astype(np.float)
        r = np.hypot(y - 20, x - 20)

        rs = np.linspace(1., 20, 7)
        ts = np.array([2, 4, 6, 8, 10, 15, 20]).astype(np.float)
        self.rings = np.column_stack(np.exp(-(r - rr) ** 2 / tt ** 2).ravel()
                                     for rr, tt in product(rs, ts))

    def _extract_rgb(self, rgb):
        r = rgb[:, :, 0].astype(np.float).ravel()
        g = rgb[:, :, 1].astype(np.float).ravel()
        rnorm = r / np.maximum(r.sum(), 1)
        gnorm = g / np.maximum(g.sum(), 1)
        result = np.hstack([np.dot(r, self.rings),
                            np.dot(g, self.rings),
                            np.dot(r, self.rings) - np.dot(g, self.rings),
                            np.dot(rnorm, self.rings),
                            np.dot(gnorm, self.rings),
                            np.dot(rnorm - gnorm, self.rings)])
        return result.reshape(1, -1)


class CompositeExtractor(Extractor):
    composite_classes = []

    def __init__(self):
        super(CompositeExtractor, self).__init__()
        self.extractors = [c() for c in self.composite_classes]

    def _extract_rgb(self, rgb):
        return np.hstack(e._extract_rgb(rgb) for e in self.extractors)


class RingWaveletCompositeExtractor(CompositeExtractor):
    composite_classes = [RingExtractor, MultiWaveletExtractor]


class RingWaveletCompressionExtractor(CompositeExtractor):
    composite_classes = [RingExtractor, MultiWaveletExtractor,
                         CompressionExtractor]


class RingWaveletCompressionStatExtractor(CompositeExtractor):
    composite_classes = [RingExtractor, MultiWaveletExtractor,
                         CompressionExtractor, RawStatsExtractor]

    def extract(self, lon, l, b, r):
        return np.hstack(e.extract(lon, l, b, r).ravel()
                         for e in self.extractors).reshape(1, -1)
