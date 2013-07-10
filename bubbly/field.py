import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .util import _sample_and_scale


class Field(object):
    def __init__(self, lon, path=None):
        self.lon = lon
        path = path or os.path.join(os.path.dirname(__file__), 'galaxy')
        self.path = path

        i4 = os.path.join(path, 'registered', '%3.3i_i4.fits' % lon)
        mips = os.path.join(path, 'registered', '%3.3i_mips.fits' % lon)
        heat = os.path.join(path, 'registered', '%3.3i_heatmap.fits' % lon)
        cat = os.path.join(path, 'registered', '%3.3i_catalog.fits' % lon)

        self.i4 = fits.getdata(i4)
        self.mips = fits.getdata(mips)
        self.wcs = WCS(fits.getheader(i4))

    def __getitem__(self, field, *slices):
        fields = dict(i4=self.i4, i4i=self.i4i, mips=self.mips,
                      mipsi=self.mipsi, heat=self.heat)
        if field not in fields:
            raise ValueError("Field must be one of %s" % (fields.keys(),))
        return fields[field][slices]

    def all_stamps(self):
        shp = self.i4.shape
        n = max(shp[0], shp[1]) / 2
        r = 40
        while r < n:
            y, x = np.mgrid[r / 2: shp[0] - r / 2: r / 5,
                            r / 2: shp[1] - r / 2: r / 5]
            y = y.ravel()
            x = x.ravel()
            lb = self.wcs.all_pix2world(np.column_stack([x, y]), 0)
            rad = r * 2. / 3600.
            for l, b in lb:
                yield (self.lon, l, b, rad)
                r = int(r * 1.25)

    def extract_stamp(self, lon, lat, size, do_scale=True, limits=None):

        lb = np.array([[lon, lat]])
        x, y = self.wcs.wcs_world2pix(lb, 0).ravel()
        x, y = map(int, [x, y])

        pixscale = 2. / 3600.
        dx = int(size / pixscale)
        lt = x - dx
        rt = x + dx
        bt = y - dx
        tp = y + dx
        mips, i4 = self.mips, self.i4
        if lt < 0 or rt >= i4.shape[1] or bt < 0 or tp >= i4.shape[0]:
            return

        sz = 2 * dx
        stride = max(int(sz / 80), 1)

        i4 = self.i4[bt:tp:stride, lt:rt:stride]
        mips = self.mips[bt:tp:stride, lt:rt:stride]
        rgb = _sample_and_scale(i4, mips, do_scale, limits)
        return rgb


class CloudField(Field):

    def __init__(self, lon):
        from cloud.bucket import sync_from_cloud
        self.lon = lon
        i4 = "%3.3i_i4.fits" % lon
        mips = "%3.3i_mips.fits" % lon

        sync_from_cloud(i4)
        sync_from_cloud(mips)

        self.i4 = fits.getdata(i4)
        self.mips = fits.getdata(mips)
        self.wcs = WCS(fits.getheader(i4))
