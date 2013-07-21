import numpy as np
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt

from bubbly.dr1 import bubble_params
from bubbly.extractors import RGBExtractor


def collage(bubbles):
    ex = RGBExtractor()
    ex.shp = (200, 200)
    images = [ex.extract(*p) for p in bubble_params(bubbles)]

    if len(images) == 3:
        return np.vstack(images)

    r, g, b = tuple(montage2d(np.array([a[:, :, i] for a in images]))
                    for i in range(3))
    return np.dstack((r, g, b)).astype(np.uint8)


def hide_axes():
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

def bubble_names(bubbles):
    params = bubble_params(bubbles)
    return ['%6.6i%+5.5i' % (p[1] * 1000, p[2] * 10000)
            for p in params]

def label(labels, w, h, nx, ny):
    labels = bubble_names(labels)
    for i in range(len(labels)):
        x = w / nx * (i % nx) + 20
        y = h / ny * (i / nx) + .8 * h / ny
        plt.annotate(labels[i], xy=(x, y), color='white',
                     backgroundcolor='#111111')


def main():

    bubbles = [13, 14, 17, 7, 8, 9, 0, 1, 3]
    snr = [12, 28, 297]
    other = other = [583, 861, 1169]#[306, 383, 583]
    pne = [361, 593, 1417]
    globule = [1535, 1597, 1600]

    b = collage(bubbles)
    s = collage(snr)
    p = collage(pne)
    g = collage(globule)
    o = collage(other)

    im = np.hstack((b, s, p, g, o))


    plt.figure(figsize=(8, 4), dpi=200)
    plt.imshow(im)

    w = im.shape[1]
    plt.axvline(3 * w / 7, color='#eeeeee', lw=3)
    plt.axvline(5 * w / 7, color='#eeeeee', lw=3)
    plt.axvline(6 * w / 7, color='#eeeeee', lw=3)


    labels = [bubbles[0], bubbles[1], bubbles[2],
              snr[0], pne[0], globule[0], other[0],
              bubbles[3], bubbles[4], bubbles[5],
              snr[1], pne[1], globule[1], other[1],
              bubbles[6], bubbles[7], bubbles[8],
              snr[2], pne[2], globule[2], other[2]]

    #label(labels, im.shape[1], im.shape[0], 7, 3)
    h, w = im.shape[0:2]
    kwargs = dict(color='white', fontsize=18, backgroundcolor='k')
    plt.annotate("a", xy=(20, h - 60), **kwargs)
    plt.annotate("b", xy=(20 + 3 * w / 7, h - 60), **kwargs)
    plt.annotate("c", xy=(20 + 5 * w / 7, h - 60), **kwargs)
    plt.annotate("d", xy=(20 + 6 * w / 7, h - 60), **kwargs)

    hide_axes()
    plt.savefig('gallery.eps')


if __name__ == "__main__":
    main()
