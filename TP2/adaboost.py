import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from decision_stump import DecisionStump


class AdaBoost(ClassifierMixin, BaseEstimator):
    X_: np.ndarray
    y_: np.ndarray
    classes_: np.ndarray

    weights_: list
    alphas_: list
    classifiers_: list
    errors_: list

    iter_acc_: list

    def __init__(self, iterations: int = 100):
        self.iterations = iterations
        self.iter_acc_ = list()

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, dtype=None, y_numeric=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.weights_ = [np.ones(X.shape[0]) / X.shape[0]]
        self.alphas_ = list()
        self.classifiers_ = list()
        self.errors_ = list()
        for i in range(self.iterations):
            self.classifiers_.append(DecisionStump())
            self.classifiers_[i].fit(self.X_, self.y_, self.weights_[i])

            prediction = self.classifiers_[i].predict(self.X_)

            self.errors_.append(np.sum(self.weights_[i] * (prediction != self.y_)) / np.sum(self.weights_[i]))
            self.alphas_.append(np.log((1 - self.errors_[i]) / self.errors_[i]))

            self.weights_.append(self.weights_[i] * np.exp(self.alphas_[i] * (prediction != self.y_)))

            self._iter_error(X, y)

        return self

    def _iter_error(self, X, y):
        self.iter_acc_.append(accuracy_score(y_true=y, y_pred=self.predict(X)))

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, dtype=None)

        predictions = np.sum(
            [
                alpha * classifier.predict(X)
                for alpha, classifier in zip(self.alphas_, self.classifiers_)
            ],
            axis=0
        ) > 0

        return predictions


if __name__ == '__main__':
    from download_data import download_data

    data, target = download_data()

    model = AdaBoost(iterations=100)
    model.fit(data, target)

    print(model.iter_acc_)
    print(accuracy_score(
        y_pred=model.predict(data),
        y_true=target,
    ))
