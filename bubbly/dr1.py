"""
Code in this file deals with the data from the
Milky Way Project First Data Release
"""
import os
import cPickle as pickle

import numpy as np
import h5py


def _on_args(catalog, row):
    """Given the row of the DR1 catalog, turn into
    (lon, lat, radius) tuple
    """
    lon, lat, pa, a, b, t, hr = catalog[row]
    pa *= -1
    return lon, lat, 1.3 * (a + t)


def _high_quality_on_locations():
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
    return [_on_args(cat, row) for row in on_id[good]]


def get_catalog():
    try:
        import cloud
        cloud.bucket.sync_from_cloud('catalog.pkl')
        return pickle.load(open('catalog.pkl'))
    except:
        pass

    if os.path.exists('catalog.pkl'):
        return pickle.load(open('catalog.pkl'))

    from MySQLdb import connect
    db = connect(host='localhost', user='beaumont', db='mwp')
    cursor = db.cursor()
    cursor.execute('select lon, lat, angle, semi_major, semi_minor, '
                   'thickness, hit_rate from clean_bubbles_anna')
    cat = np.array([map(float, row) for row in cursor.fetchall()])

    with open('catalog.pkl', 'w') as outfile:
        pickle.dump(cat, outfile)

    return cat


def on_stamp_params():
    """Iterate over high quality + examples,
    yield tuples of (field_lon, lcen, bcen, r)

    These can be fed to Extractor objects
    """
    for l, b, r in _high_quality_on_locations():
        yield int(np.round(l)), l, b, r
