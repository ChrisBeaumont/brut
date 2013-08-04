import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['axes.facecolor'] = '#ffffff'
from scipy.ndimage import label

from bubbly.cluster import merge
from bubbly.field import get_field
from bubbly.util import scale
from bubbly.dr1 import bubble_params


def plot_stamps(stamps, **kwargs):
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('edgecolor', 'b')
    kwargs.setdefault('alpha', .1)
    label = kwargs.pop('label', None)

    ax = plt.gca()
    for s in stamps:
        r = Rectangle((s[1] - s[-1], s[2] - s[-1]),
                      width = 2 * s[-1], height = 2 * s[-1], **kwargs)
        ax.add_patch(r)

    if label is not None:
        plt.plot([np.nan], [np.nan], '-', color = kwargs['edgecolor'],
                 label=label)

def main():

    data = json.load(open('../models/l035_scores.json'))

    stamps = np.array(data['stamps'])
    scores = np.array(data['scores'])

    l = stamps[:, 1]
    b = stamps[:, 2]

    good = (scores > .1) & (l < 35.17) & (l > 34.9) & (b > -.9) & (b < -0.6)

    stamps = stamps[good]
    scores = scores[good]

    merged = merge(stamps, scores)
    mwp = np.array(bubble_params())
    mwp  = mwp[(mwp[:, 1] < 35.3) & (mwp[:, 1] > 35)]

    f = get_field(35)
    bad = f.mips == 0
    g = scale(f.i4, limits=[30, 99.8])
    r = scale(f.mips, limits=[30, 99.7])
    r[bad] = 255
    b = r * 0

    im = np.dstack((r, g, b))

    plt.figure(dpi=200, tight_layout=True)
    plt.imshow(im, extent=[36, 34, -1, 1], interpolation="bicubic")

    plot_stamps(merged, edgecolor='#7570b3', linewidth=2, label='Brut')
    plot_stamps(mwp, edgecolor='#e7298a', linewidth=2, label='MWP')

    plt.xlim(35.2, 35)
    plt.ylim(-.825, -.625)
    plt.legend(loc='upper right')

    plt.xlabel("$\ell$ ($^\circ$)")
    plt.ylabel("b ($^\circ$)")

    plt.savefig('cluster_confusion.eps')


if __name__ == "__main__":
    main()
