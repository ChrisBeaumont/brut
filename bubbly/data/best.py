import cPickle as pickle
import os

from skimage.io import imsave
import numpy as np

from make_hiq import filter
from bubbly.extractors import RGBExtractor


def main():

    ex = RGBExtractor()
    ex.shp = (100, 100)

    data = filter(hit_rate=0.2)

    params = {}
    for i, d in enumerate(data):
        d1 = list(d)
        d2 = list(d)
        d2[-1] *= 2

        l = int(np.round(d[0]))
        d1 = [l] + d1
        d2 = [l] + d2

        im = np.hstack((ex(*d1), ex(*d2)))
        pth = "vet_%4.4i.png" % i
        params[pth] = d
        pth = os.path.join('best', pth)
        imsave(pth, im)

    pth = os.path.join('best', 'data.pkl')
    with open(pth, 'w') as outfile:
        pickle.dump(params, outfile)


if __name__ == "__main__":
    main()
