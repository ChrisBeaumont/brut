import json

import numpy as np
import matplotlib.pyplot as plt

from bubbly.extractors import RGBExtractor
from bubbly.dr1 import bubble_params

def hide_axes():
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().axis('off')

def ex(params):
    rgb = RGBExtractor()
    rgb.shp = (200, 200)

    p = list(params)
    p[-1] *= 1.5
    return rgb.extract(*p)


def main():
    data = json.load(open('../models/training_data_0.json'))
    data = data['neg']


    images = [ex(data[i]) for i in [40, 41, -3, -1]]
    dx = images[0].shape[0]

    images = np.hstack(np.vstack(images[i:i+2]) for i in [0, 2])

    plt.imshow(images)

    kw = {'color':'w', 'lw':4}
    plt.axhline(dx, **kw)
    plt.axvline(dx, **kw)

    kw = {'ha':'center', 'va':'bottom', 'fontsize':14}
    plt.annotate("Random", xy=(dx / 2, 2 * dx - 40), color='white', **kw)
    plt.annotate("Random", xy=(dx / 2, dx - 40), color='white', **kw)
    plt.annotate("Hard", xy=(3 * dx / 2, 2 * dx - 40), color='k', **kw)
    plt.annotate("Hard", xy=(3 * dx / 2, dx - 40), color='k', **kw)
    plt.title("Negative Training Examples")
    hide_axes()

    plt.savefig('bootstrap_neg.eps')



if __name__ == "__main__":
    main()
