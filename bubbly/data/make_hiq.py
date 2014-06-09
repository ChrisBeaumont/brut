import numpy as np
import h5py
import cPickle as pickle

from bubbly.dr1 import get_catalog, _on_args

def filter(hit_rate=0.3):
    cat = get_catalog()
    infile = 'stamps2.h5'
    on_id = np.array(map(int, h5py.File(infile, 'r')['on'].keys()))

    rgbs = h5py.File(infile, 'r')['on'].values()

    #has green channel...
    def has_green(rgb):
        return (rgb[:, :, 1] == 0).mean() < 0.3

    good = np.array([has_green(r) for r in rgbs], dtype=np.bool)

    #...and hit rate > 0.3
    good = good & (cat[on_id, -1] > 0.3)
    result = [_on_args(cat, row) for row in on_id[good]]
    return result

def main():
    pth = 'hiq.pkl'

    result = filter()

    with open(pth, 'w') as outfile:
        pickle.dump(result, outfile)

if __name__ == "__main__":
    main()
