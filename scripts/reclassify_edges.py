"""
The first round of classifications used 1deg^2 mosaics, and were
unable to yield valid classifications near longitude edges. This
script re-classifies the original fields without classification
scores, using (2deg, 1deg) mosaics. This should yield new
classifications near longitude edges
"""
from h5py import File
import numpy as np

from bubbly.model import ModelGroup

def redo(field):
    """
    Reclassify nan-scores from a single field in full_search_old,
    write to new files in full_search

    Parameters
    ----------
    field : integer
        Longitude of field to reclassify
    """
    result = []
    old = '../data/full_search_old/%3.3i.h5' % field
    new = '../data/full_search/%3.3i.h5' % field

    model = ModelGroup.load('../models/full_classifier.dat')

    with File(old) as infile:
        stamps, scores = infile['stamps'][:], infile['scores'][:]
        redo = ~np.isfinite(infile['scores'])

    new_scores = model.decision_function(stamps[redo])
    print np.isfinite(new_scores).sum()

    scores[redo] = new_scores

    with File(new, 'w') as outfile:
        outfile.create_dataset('stamps', data=stamps, compression=9)
        outfile.create_dataset('scores', data=scores, compression=9)


if __name__ == "__main__":
    from glob import glob
    from multiprocessing import Pool

    p = Pool()

    files = glob('../data/full_search_old/*h5')
    fields = map(int, [f.split('/')[-1].split('.')[0] for f in files])

    p.map(redo, fields)
