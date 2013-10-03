import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['axes.facecolor'] = '#ffffff'

from bubbly.cluster import merge
from bubbly.field import get_field
from bubbly.util import scale


def plot_stamps(stamps, **kwargs):
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('edgecolor', 'b')
    label = kwargs.pop('label', None)

    kw2 = kwargs.copy()
    kw2['edgecolor'] = 'k'
    kw2['linewidth'] = 2.0

    ax = plt.gca()
    for s in stamps:
        s[2] += np.random.normal(0, .003)
        s[1] += np.random.normal(0, .003)

        r = Ellipse((s[1], s[2]),
                      width = 2 * s[-1], height = 2 * s[-1], **kwargs)
        r2 = Ellipse((s[1], s[2]),
                      width = 2 * s[-1], height = 2 * s[-1], **kw2)

        ax.add_patch(r2)
        ax.add_patch(r)

    if label is not None:
        plt.plot([np.nan], [np.nan], '-', color = kwargs['edgecolor'],
                 label=label)

def main():
    np.random.seed(42)

    data = json.load(open('../models/l035_scores.json'))

    stamps = np.array(data['stamps'])
    scores = np.array(data['scores'])

    l = stamps[:, 1]
    b = stamps[:, 2]

    good = (scores > .1) & (l < 34.8) & (l > 34.6) & (b > -.4) & (b < -0.2)
    assert good.sum() > 0

    stamps = stamps[good]
    scores = scores[good]

    merged, ms = merge(stamps, scores)

    f = get_field(35)
    g = scale(f.i4, limits=[70, 99])
    r = scale(f.mips, limits=[70, 99])
    b = r * 0

    im = np.dstack((r, g, b))

    plt.figure(dpi=200, tight_layout=True)
    plt.imshow(im, extent=[36, 34, -1, 1], interpolation="bicubic")

    plot_stamps(stamps, linewidth=1, edgecolor='white', label='Raw',
                alpha=1)
    plot_stamps(merged, edgecolor='red', alpha=1, linewidth=2,
                label='Merged')

    plt.xlim(34.795, 34.695)
    plt.ylim(-.365, -.265)


    plt.xlabel("$\ell$ ($^\circ$)")
    plt.ylabel("b ($^\circ$)")
    leg = plt.legend(loc='upper left', frameon=False)
    for text in leg.get_texts():
        text.set_color('white')

    plt.savefig('cluster.eps')


if __name__ == "__main__":
    main()
