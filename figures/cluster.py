import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['axes.facecolor'] = '#ffffff'

from bubbly.cluster import merge
from bubbly.field import get_field
from bubbly.util import scale


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

    good = (scores > .1) & (l < 34.8) & (l > 34.6) & (b > -.4) & (b < -0.2)
    assert good.sum() > 0

    stamps = stamps[good]
    scores = scores[good]

    merged = merge(stamps, scores)

    f = get_field(35)
    g = scale(f.i4, limits=[70, 99])
    r = scale(f.mips, limits=[70, 99])
    b = r * 0

    im = np.dstack((r, g, b))

    plt.figure(dpi=200, tight_layout=True)
    plt.imshow(im, extent=[36, 34, -1, 1], interpolation="bicubic")

    plot_stamps(merged, edgecolor='#7570b3', alpha=1, linewidth=5,
                label='Merged')
    plot_stamps(stamps, linewidth=1, edgecolor='#cccccc', label='Raw')

    plt.xlim(34.795, 34.695)
    plt.ylim(-.365, -.265)


    plt.xlabel("$\ell$ ($^\circ$)")
    plt.ylabel("b ($^\circ$)")
    plt.legend(loc='upper right')
    plt.savefig('cluster.eps')


if __name__ == "__main__":
    main()
