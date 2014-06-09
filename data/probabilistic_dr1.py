import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from astropy.table import Table
from astropy.io import ascii

from bubbly.dr1 import get_catalog

def xmatch(x, y):
    nn = NearestNeighbors(n_neighbors=1).fit(x)
    ind = nn.kneighbors(y, return_distance=False)
    return ind.ravel()

def pbubble(mlscore, hr):
    #Parameters for Logistic fit determined in Expert_votes.ipynb
    clf = LogisticRegression()
    clf.intercept_ = np.array([ 1.868260])
    clf.coef_ = np.array([[1.539796]])

    # zsacling and combining. see Expert_votes.ipynb
    s = (hr - .2143) / .11379 + (mlscore - .10044) / .5893

    good = np.isfinite(s)

    result = np.zeros_like(mlscore) * np.nan
    result[good] = clf.predict_proba(s[good].reshape(-1, 1))[:, 1]
    return result

def main():

    cat = get_catalog()
    lon, lat, pa, a, b, t, hr = cat.T
    cat_stamps = np.column_stack((lon, lat, 1.3 * (a + t)))

    scores = json.load(open('../models/bubble_scores.json'))
    stamps = np.array(scores['params'])
    stamps = stamps[:, 1:]
    scores = np.array(scores['scores'])

    ind = xmatch(stamps, cat_stamps)
    assert (stamps[ind] == cat_stamps).all()

    scores = pbubble(scores[ind], hr)

    print (scores > .5).sum(), (scores > .8).sum()

    table = Table([lon, lat, pa, a, b, t, scores],
                  names=('lon', 'lat', 'pa', 'a', 'b', 't', 'prob'))

    with open('pdr1.csv', 'w') as outfile:
        ascii.write(table, outfile, delimiter=',')



if __name__ == "__main__":
    main()
