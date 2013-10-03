import json

from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt


def main():

    labels = json.load(open('../models/benchmark_scores.json'))
    off = labels['off_score']

    on = json.load(open('../models/bubble_scores.json'))['scores']

    yp = np.array(on + off)
    y = np.array([1] * len(on) + [0] * len(off))

    pars = np.array(labels['on'] + labels['off'])
    for i in [0, 1, 2]:
        mask = pars[:, 0] % 3 == i
        fp, tp, _ = roc_curve(y[mask], yp[mask])
        plt.plot(fp, tp, label = 'Zone %i' % (i + 1))


    plt.xlim(0, .002)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.gcf().set_tight_layout(True)
    plt.savefig('roc_all.eps')

if __name__ == "__main__":
    main()
