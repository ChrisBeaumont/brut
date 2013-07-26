import random
import json

from bubbly.model import ModelGroup

def locations():
    random.seed(42)
    data = json.load(open('../models/bootstrapped_labels.json'))
    on = sorted(data['on_params'])
    off = sorted(random.sample(data['off_params'], 50000))
    return on, off


def main():

    model = ModelGroup.load('../models/full_classifier.dat')
    on, off = locations()

    result = {'on': on, 'off': off}
    result['on_score'] = model.decision_function(on).tolist()
    result['off_score'] = model.decision_function(off).tolist()

    with open('../models/benchmark_scores.json', 'w') as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    main()
