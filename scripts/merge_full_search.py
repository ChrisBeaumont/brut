from glob import glob
import os

import h5py
import numpy as np
from astropy.table import Table

from bubbly.cluster import merge, xmatch

def above_thresh(thresh):
    result = []
    scores = []
    for file in glob('../data/full_search/*.h5'):
        with h5py.File(file, 'r') as data:
            good = data['scores'][...] > thresh
            result.append(data['stamps'][...][good])
            scores.append(data['scores'][...][good])
            print file, good.sum()
    return np.vstack(result), np.hstack(scores)

def chunked_merge(stamps, scores):
    stamps[stamps[:, 1] > 180, 1] -= 360
    lon = stamps[:, 1]

    ostamps, oscores = [], []
    for lcen in np.arange(lon.min(), lon.max() + 1, 1):
        good = np.abs(lon - lcen) < 1
        if good.sum() == 0:
            continue
        st, sc = merge(stamps[good], scores[good])
        good = np.abs(st[:, 1] - lcen) < .5
        if good.sum() == 0:
            continue
        ostamps.append(st[good])
        oscores.append(sc[good])
        print lcen, good.sum()

    result = merge(np.vstack(ostamps), np.hstack(oscores))
    result[0][result[0][:, 1] < 0, 1] += 360
    return result

def write_catalog(stamps, scores, outfile):
    t = Table([stamps[:, 1], stamps[:, 2], stamps[:, 3], scores],
               names = ['lon', 'lat', 'rad', 'score'])
    t.write(outfile, format='ascii', delimiter=',')


def main():
    thresh = 0.2
    stamps, scores = above_thresh(thresh)
    print "Number of fields above %f: %i" % (thresh, len(scores))

    merged, mscores = chunked_merge(stamps, scores)
    print "Number of fields after merging: %i" % len(mscores)

    write_catalog(merged, mscores, '../data/full_search.csv')

if __name__ == '__main__':
    main()
