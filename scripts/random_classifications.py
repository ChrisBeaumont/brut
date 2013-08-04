import random
import json

from bubbly.model import ModelGroup
from bubbly.dr1 import UnrestrictedLocationGenerator


def locations():
    random.seed(42)
    lg = UnrestrictedLocationGenerator()
    fields = lg.random_iterator()
    return sorted([next(fields) for _ in range(50000)])


def main():

    model = ModelGroup.load('../models/full_classifier.dat')
    loc = locations()
    result = {'stamps': loc}
    result['scores'] = model.decision_function(loc).tolist()

    with open('../models/random_scores.json', 'w') as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    main()
