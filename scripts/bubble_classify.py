import json
from bubbly.model import ModelGroup
from bubbly.dr1 import bubble_params


def main():

    model = ModelGroup.load('../models/full_classifier.dat')
    ind = np.argsort(bubble_params())
    bubbles = sorted(bubble_params())
    scores = model.decision_function(bubbles)



    result = {'params': bubbles, 'scores': scores.tolist(),
              'ind': ind}

    with open('../models/bubble_scores.json', 'w') as outfile:
        json.dump(result, outfile)




if __name__ == "__main__":
    main()
