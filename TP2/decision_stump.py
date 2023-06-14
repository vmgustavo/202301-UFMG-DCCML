from typing import Any
from operator import eq
from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class DecisionStump(ClassifierMixin, BaseEstimator):
    X_: Any
    y_: Any
    classes_: Any
    best_attr_test_: Any

    def __init__(self):
        pass

    def fit(self, X, y, weights):
        X, y = check_X_y(X, y, weights, dtype=None, y_numeric=True)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        entropy_init = self._entropy_step(self.y_, weights)

        tests = [
            lambda x: eq(x, 'x'),
            lambda x: eq(x, 'o'),
            lambda x: eq(x, 'b'),
        ]

        decisions = list()
        decision = namedtuple('decision', ['attr', 'test', 'gain', 'output'])
        for attr in range(self.X_.shape[1]):
            for test in tests:
                tested = test(self.X_[:, attr])
                group0 = self.y_[np.where(tested)]
                group0_weights = weights[np.where(tested)]
                group1 = self.y_[np.where(~tested)]
                group1_weights = weights[np.where(~tested)]

                entropy_new = (
                    group0.shape[0] / self.y_.shape[0] * self._entropy_step(group0, group0_weights)
                    + group1.shape[0] / self.y_.shape[0] * self._entropy_step(group1, group1_weights)
                )

                info_gain = entropy_init - entropy_new

                _, counts = np.unique(group0, return_counts=True)

                decisions.append(decision(attr, test, info_gain, np.argmax(counts)))

        self.best_attr_test_ = sorted(decisions, key=lambda x: x.gain, reverse=True)[0]

        return self

    @staticmethod
    def _entropy(a: int, b: int):
        freq = a / (a + b)
        return -1 * freq * np.log(freq)

    def _entropy_step(self, arr: np.ndarray, weights):
        total_0 = int(np.sum(weights[np.where(arr == 0)]))
        total_1 = int(np.sum(weights[np.where(arr == 1)]))

        return (
            self._entropy(total_0, total_1)
            + self._entropy(total_1, total_0)
        )

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, dtype=None)
        attr, test, _, output = self.best_attr_test_

        if bool(output):
            return test(X[:, attr]).astype('int')
        else:
            return (~test(X[:, attr])).astype('int')


if __name__ == '__main__':
    from download_data import download_data

    data, target = download_data()

    model = DecisionStump()
    model.fit(data, target, np.ones(target.shape[0]))
    print(accuracy_score(
        y_pred=model.predict(data),
        y_true=target,
    ))
