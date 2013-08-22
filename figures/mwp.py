import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import aplpy

def main():
    fig = plt.figure(tight_layout=True, dpi=800)
    im1 = 'GLM_04393+0006_mosaic_I24M1.jpg'
    im2 = 'GLM_04130-0013_mosaic_I24M1.jpg'
    im1 = imread(im1)[::-1, ...]
    im2 = imread(im2)[::-1, ...]

    plt.subplot(211)
    plt.imshow(im1, interpolation='bicubic')

    plt.plot([.05 * 3600/3.375, .25 * 3600/3.375], [50, 50], 'white')
    plt.xlim(0, 800)
    plt.ylim(0, 400)

    plt.annotate("$\ell=43.93^\circ$  $b=0.06^\circ$",
                 xy=(12, .9 * im1.shape[0]),
                 color='white')
    plt.annotate("$0.2^\circ$", xy=(.15 * 3600 / 3.375, 80),
                 color='white', ha='center')

    plt.xticks([])
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(im2, interpolation='bicubic')
    plt.annotate("$\ell=41.06^\circ$  $b=0.17^\circ$",
                 xy=(12, .9 * im1.shape[0]),
                 color='white')
    plt.plot([.05 * 3600/6.75, .25 * 3600/6.75], [50, 50], 'white')
    plt.annotate("$0.2^\circ$", xy=(.15 * 3600 / 6.75, 80),
                 color='white', ha='center')
    plt.xlim(0, 800)
    plt.ylim(0, 400)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('mwp.eps')

if __name__ == '__main__':
    main()
