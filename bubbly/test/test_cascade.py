from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
import numpy as np
import pytest


from ..cascade import CascadedBooster, _recall_bias, _fpos_bias
from ..util import false_pos


def test_recall_bias():

    df = np.arange(5)

    for thresh in [0, .1, .2, .21, .5, 1]:
        bias = _recall_bias(df, thresh)
        assert (df >= bias).mean() >= thresh


def test_fpos_bias():
    df = np.arange(5)

    for thresh in [0, .1, .2, .21, .5, 1]:
        bias = _fpos_bias(df, thresh, 5)
        assert (df >= bias).mean() <= thresh


class TestCascade(object):

    def setup_method(self, method):
        base = GradientBoostingClassifier(n_estimators=2)
        self.clf = CascadedBooster(base_clf=base)

        n_samples = 500
        np.random.seed(42)
        X, Y = make_moons(n_samples=n_samples, noise=.05)

        self.X = X
        self.Y = Y
        self.clf.fit(X, Y)

    def teardown_method(self, method):
        self.check_clf(self.clf, self.X, self.Y)

    def test_multi_layer(self):
        assert len(self.clf.estimators_) > 1

    def check_recall_constraints(self, clf, x, y):
        #recall is at least as good as required
        recall_i = clf.layer_recall
        for i, yp in enumerate(clf.staged_predict(x), 1):
            assert recall_score(y, yp) >= (recall_i ** i)
        np.testing.assert_array_equal(yp, clf.predict(x))

    def check_fpos_constraints(self, clf, x, y):
        # if classifier converged, false pos rate on training data <= false_pos
        if clf.converged:
            assert false_pos(y, clf.predict(x)) <= clf.false_pos

    def check_staged_decision(self, clf, x, y):
        for i, yp in enumerate(clf.staged_decision_function(x)):
            good = yp > 0
            #df for positive examples == df for last stage
            expect = clf.estimators_[i].decision_function(x[
                                                          good]) - clf.bias_[i]
            np.testing.assert_array_equal(yp[good].ravel(), expect.ravel())

        np.testing.assert_array_equal(clf.decision_function(x), yp)

    def check_staged_predict(self, clf, x, y):
        #staged predict and decision function are consistent
        from itertools import izip
        for df, yp in izip(clf.staged_decision_function(x),
                           clf.staged_predict(x)):
            np.testing.assert_array_equal((df >= 0).ravel(), (yp > 0).ravel())

    def check_clf(self, clf, x, y):
        self.check_recall_constraints(clf, x, y)
        self.check_fpos_constraints(clf, x, y)
        self.check_staged_decision(clf, x, y)
        self.check_staged_predict(clf, x, y)

    def test_recall_stages(self):
        self.check_recall_constraints(self.clf, self.X, self.Y)

        self.clf.layer_recall = 1.0
        self.clf.fit(self.X, self.Y)
        self.check_recall_constraints(self.clf, self.X, self.Y)
        assert recall_score(self.clf.predict(self.X), self.Y) == 1

    def test_fit_converge(self):
        self.clf.false_pos = 0.5
        self.clf.fit(self.X, self.Y)
        assert self.clf.converged

    def test_fit_maxiter(self):
        self.clf.false_pos = 0
        self.clf.layer_recall = .6

        self.clf.max_layers = 4

        np.random.shuffle(self.Y)
        self.clf.fit(self.X, self.Y)

        assert not self.clf.converged
        assert len(self.clf.bias_) == 4

    def test_impossible_filter(self):
        self.clf.layer_recall = 1
        self.clf.false_pos = 0
        np.random.shuffle(self.Y)

        self.clf.fit(self.X, self.Y)
        assert not self.clf.converged
        assert len(self.clf.bias_) < 4

    def test_separable_fit(self):
        self.clf.layer_recall = .5
        self.clf.false_pos = .5

        self.X = np.arange(10).reshape(-1, 1)
        self.Y = (self.X.ravel() > 5).astype(np.int)

        self.clf.fit(self.X, self.Y)
        assert self.clf.converged
        assert len(self.clf.estimators_) == 1

        assert recall_score(self.Y, self.clf.predict(self.X)) == 1
        assert false_pos(self.Y, self.clf.predict(self.X)) == 0

    def test_add_layer(self):
        recall = recall_score(self.Y, self.clf.predict(self.X))
        fpos = false_pos(self.Y, self.clf.predict(self.X))
        n_layers = len(self.clf.estimators_)

        self.clf.add_cascade_layer(self.X, self.Y)
        assert len(self.clf.estimators_) == n_layers + 1
        yp = self.clf.predict(self.X)
        assert false_pos(self.Y, yp) <= fpos
        assert recall_score(self.Y, yp) >= recall * self.clf.layer_recall

    def test_predict_before_fit(self):

        clf = CascadedBooster()
        for func in [clf.staged_predict,
                     clf.staged_decision_function,
                     clf.predict,
                     clf.decision_function,
                     ]:
            with pytest.raises(ValueError) as exc:
                func()
