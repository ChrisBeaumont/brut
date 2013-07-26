import json

from bubbly.model import ModelGroup
from bubbly.field import get_field


def main():

    model = ModelGroup.load('../models/full_classifier.dat')

    f = get_field(35)
    stamps = list(f.small_stamps())
    stamps = [s for s in stamps if s[1] > 34.5 and s[1] < 35.5]

    df = model.cloud_decision_function(stamps, workers=100)
    result = {'stamps': stamps, 'scores': df.tolist()}

    with open('../models/l035_small_scores.json', 'w') as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    main()
