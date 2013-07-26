"""
This module defines the BubblyViewer class,
which wraps pyds9 to look at bubble catalogs.

Usage
-----

bv = BubblyViewer()

# open ds9, load l=35 data
bv.load_longitude(35)

# draw bubbles as circles
bv.outline(bubble_params(), color='green')

# pan/zoom to 5th entry
bv.look_at(bubble_params()[5])

#delete annotations
bv.clear()
"""

import ds9
import os
from bubbly.field import Field

__all__ = ['BubblyViewer']

class BubblyViewer(object):
    def __init__(self):
        self.ds9 = None

    def start(self):
        """Start ds9 if needed"""
        if self.ds9 is None:
            self.ds9 = ds9.ds9()

    def load_longitude(self, lon):
        """Load the image data associated with a given longitude"""
        self.start()

        f = Field(lon)
        g = os.path.join(f.path, 'registered', '%3.3i_i4.fits' % lon)
        r = os.path.join(f.path, 'registered', '%3.3i_mips.fits' % lon)
        self.ds9.set('frame delete')
        self.ds9.set('frame new rgb')
        self.ds9.set('rgb red')
        self.ds9.set('file %s' % r)
        self.ds9.set('rgb green')
        self.ds9.set('file %s' % g)

        self._set_zscale()
        self._align_galactic()

    def _set_zscale(self):
        self.ds9.set('rgb red')
        self.ds9.set('scale asinh')
        self.ds9.set('rgb green')
        self.ds9.set('scale asinh')

    def _align_galactic(self):
        self.ds9.set('wcs galactic')
        self.ds9.set('wcs skyformat degrees')

    def look_at(self, params):
        """Center on a specific bubble

        Parameters
        ----------
        params: Tuple
            A stamp description tuple of the form
            (lon_field, lon, lat, radius)

            This is returned by, e.g., bubble_params(),
            Field.all_stamps(), etc.
        """
        if self.ds9 is None:
            self.load_longitude(params[0])

        l, b = params[1:3]
        r = params[-1]

        #this is a hacky guess
        zoom = 2 / 3600. / r * 500

        self.ds9.set('pan to {l} {b} galactic'.format(l=l, b=b))
        self.ds9.set('zoom to %f' % zoom)

    def outline(self, params, color='blue'):
        """
        Display a list of stamps as circular regions

        Parameters
        ----------
        params : tuple, or list of tuples
            Stamp descriptions of the form (lon_field, lon, lat, radius)

        color : string
            A ds9-recognized color to use as the region outline
        """
        if not hasattr(params[0], '__len__'):
            params = [params]

        for i, p in enumerate(params):
            l, b, r = p[1:]
            self.ds9.set('regions',
                         'galactic; circle(%f,%f,%f)#color=%s text="%s"' %
                         (l, b, r, color, i))

    def clear(self):
        """
        Remove all annotations
        """
        self.ds9.set('regions delete all')
