import json
import cPickle as pickle

from bubbly.model import Model, ModelGroup
from bubbly.extractors import MultiViewExtractor, ManyManyExtractors
from bubbly.dr1 import WideLocationGenerator
from bubbly.wiserf import WiseRF


def make_model(mod3):
    params = {'max_features': 'auto',
              'min_samples_split': 4,
              'n_jobs': 2,
              'criterion': 'infogain',
              'n_estimators': 800}
    ex = MultiViewExtractor(ManyManyExtractors())
    loc = WideLocationGenerator(mod3)
    clf = WiseRF(**params)
    return Model(ex, loc, clf)


def train_model(model, mod3):
    data = json.load(open('../models/training_data_%i.json' % mod3))
    model.fit(data['pos'], data['neg'])
    return model


def main():

    models = [train_model(make_model(i), i) for i in [0, 1, 2]]
    mg = ModelGroup(*models)
    mg.save('../models/full_classifier.dat')

if __name__ == "__main__":
    main()
