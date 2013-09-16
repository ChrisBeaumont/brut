import json

from bubbly.model import ModelGroup
from bubbly.field import get_field


def main():

    model = ModelGroup.load('../models/full_classifier.dat')

    f = get_field(305)
    stamps = sorted(list(f.all_stamps()))

    df = model.cloud_decision_function(stamps, workers=100)
    result = {'stamps': stamps, 'scores': df.tolist()}

    with open('../models/l305_scores.json', 'w') as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    main()
