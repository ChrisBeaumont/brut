import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from bubbly.extractors import RGBExtractor
from bubbly.cluster import merge
from bubbly.field import get_field

class CustomExtractor(RGBExtractor):

    def extract(self, lon, l, b, r):
        shp = self.shp
        return get_field(lon).extract_stamp(l, b, r, limits=[30, 98], shp=shp)


def hide_axes():
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

def collage(ims, nx, ny):

    shp = ims[0].shape
    result = np.zeros((shp[0] * ny, shp[1] * nx, 3), dtype=np.uint8)

    for i, im in enumerate(ims):
        x = i % nx
        y = i / nx
        result[shp[0] * y: shp[0] * (y + 1),
               shp[1] * x: shp[1] * (x + 1), :] = im[::-1]

    plt.imshow(result, origin='upper', extent=[0, nx, ny, 0],
               interpolation='bicubic')


def trace_groups(groups, colors, nx, ny):

    i0 = 0

    for g, c in zip(groups, colors):
        for i in range(len(g)):
            x = (i0 + i) % nx
            y = (i0 + i) / nx
            r = Rectangle((x, y), 1, 1, fc='none', ec=c)
            plt.gca().add_patch(r)
        i0 += len(g)

def ex(params):
    rgb = CustomExtractor()
    rgb.shape = (200, 200)
    p = list(params)
    p[-1] *= 1.5
    return rgb.extract(*p)

def main():

    data = json.load(open('../models/l035_scores.json'))

    stamps = np.array(data['stamps'])
    scores = np.array(data['scores'])

    unmerged = stamps[scores > .2]
    merged = merge(unmerged, scores[scores > 0.2])


    unambig = [1, 3, 4, 6, 7, 11, 12, 13, 15, 16, 17, 18, 19]
    ambig = [2, 9]
    new = [14]
    fp = [0, 5]
    neb = [8, 10]
    groups = [unambig, ambig, new, fp, neb]


    ims = [ex(merged[i]) for g in groups for i in g]

    plt.figure(dpi=400, tight_layout=True)
    collage(ims, 5, 4)
    trace_groups(groups, 'rgbcm', 5, 4)

    hide_axes()
    plt.savefig('l35.eps')


if __name__ == "__main__":
    main()
