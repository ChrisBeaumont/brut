"""
Grab IRAC bands 3,4 and MIPS mosaics from the web
"""
import os
from time import time
from urllib import urlopen
from multiprocessing import Pool
from functools import partial

def get_url(url, fname, target_dir='galaxy/raw/'):
    """ Save a file from the internet

    Parameters
    ----------
    url : The url to download
    fname : The filename to save to
    target_dir : The directory to save into. Default= galaxy/raw
    """
    outfile = os.path.join(target_dir + fname)
    if os.path.exists(outfile) and (os.path.getsize(outfile) > 0):
        return

    print '+%s' % (url + fname)
    data = urlopen(url + fname)

    if data.getcode() != 200:
        print '%s returned status code %i' % (url + fname, data.getcode())
        return

    t0 = time()
    with open(outfile, 'wb') as out:
        out.write(data.read())
    t1 = time()
    print 'Downloaded in %i seconds' % (t1 - t0)

def get_glimpse(folder, lon, survey='I'):
    """Download GLIMPSE images I3 and I4 for a single longitude

    Parameters
    ----------
    folder : Sub-directory on IPAC website (e.g. 'GLON_10-30')
    lon : Longitude tile to grab
    survey : Optional survey ('I', 'II')
    """

    base_url = 'http://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/images/%s/1.2_mosaics_v3.5/' % survey
    i3 = 'GLM_%3.3i00+0000_mosaic_I3.fits' % lon
    i4 = 'GLM_%3.3i00+0000_mosaic_I4.fits' % lon
    get_url(base_url + folder, i3)
    get_url(base_url + folder, i4)

def get_mips(lon):
    """ Download MIPS images at b = +/- .5deg

    Parameters
    ----------
    lon : Longitude to grab
    """

    base_url = 'http://irsa.ipac.caltech.edu/data/SPITZER/MIPSGAL/images/mosaics24/'

    pos = 'MG%3.3i0p005_024.fits' % lon
    neg = 'MG%3.3i0n005_024.fits' % lon

    get_url(base_url, pos)
    get_url(base_url, neg)


def get_all(threads=5):
    """ Grab all the data

    Parameters
    ----------
    Threads : how many worker threads to use
    """
    p = Pool(threads)

    lon = range(12, 30, 3)
    folder = 'GLON_10-30/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(30, 52, 3)
    folder = 'GLON_30-53/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(33, 67, 3)
    folder = 'GLON_53-66/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(294, 310, 3)
    folder = 'GLON_284_295-310/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(312, 328, 3)
    folder = 'GLON_310-330/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(330, 349, 3)
    folder = 'GLON_330-350/'
    p.map(partial(get_glimpse, folder), lon)

    lon = range(0, 10, 3)
    folder = ''
    p.map(partial(get_glimpse, folder, survey='II'), lon)

    lon = range(351, 358, 3)
    folder = ''
    p.map(partial(get_glimpse, folder, survey='II'), lon)

    #XXX GLIMPSE 3D

    lon = range(70)
    p.map(get_mips, lon)

    lon = range(292, 361)
    p.map(get_mips, lon)


if __name__ == "__main__":
    get_all()
