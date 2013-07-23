import json

import numpy as np
import matplotlib.pyplot as plt

from bubbly.extractors import RGBExtractor
from bubbly.dr1 import bubble_params

def hide_axes():
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

def ex(params):
    rgb = RGBExtractor()
    rgb.shp = (100, 100)

    p = list(params)
    p[-1] *= 1.5
    return rgb.extract(*p)

def closest_bubble(p):

    bubbles = np.array(bubble_params())
    l0, b0 = bubbles[:, 1], bubbles[:, 2]

    d = np.hypot(l0 - p[1], b0 - p[2])
    ind = np.argmin(d)
    return bubbles[ind], d[ind]

def main():

    labels = json.load(open('../models/benchmark_scores.json'))

    ind = np.argsort(labels['off_score'])[::-1]
    scores = [labels['off_score'][i] for i in ind[:9]]
    images = [ex(labels['off'][i]) for i in ind[:9]]

    for i in range(9):
        p = labels['off'][ind[i]]
        b, d = closest_bubble(p)
        print "Offset: %0.2f\t radii: %0.2f %0.2f" % (d, p[-1], b[-1])

    ims = np.vstack(np.hstack(images[i:i+3]) for i in [0, 3, 6])
    plt.imshow(ims, origin='upper')

    dx = images[0].shape[0]
    kw = {'color': 'white'}
    plt.axhline(dx, **kw)
    plt.axhline(dx * 2, **kw)

    plt.axvline(dx, **kw)
    plt.axvline(dx * 2, **kw)

    for i in range(9):
        x = dx * (i % 3) + dx / 10
        y = dx * (i / 3) + 9.5 * dx / 10
        plt.annotate("%0.2f" % scores[i], xy=(x, y), color='white')


    hide_axes()
    plt.savefig('fp.eps')


if __name__ == "__main__":
    main()
