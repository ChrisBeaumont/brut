import os

import PyWiseRF
import cloud
import numpy as np

if cloud.running_on_cloud():
    os.environ['WISERF_ROOT'] = '/home/picloud/WiseRF-1.5.9-linux-x86_64-rc2'


def test():
    x = np.random.random((5, 5))
    y = np.array([1, 1, 1, 0, 0])
    clf = WiseRF().fit(x, y)
    return clf

class WiseRF(PyWiseRF.WiseRF):
    def decision_function(self, x):
        p = self.predict_proba(x)
        return p[:, 1] - p[:, 0]
