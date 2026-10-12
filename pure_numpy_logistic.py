"""Educational multinomial logistic regression in NumPy (without scikit-learn).

Deterministic, scaled features, L2-penalized cross-entropy, Adam optimization.
It is a fallback implementation, not an exact numerical clone of sklearn.
"""

import numpy as np


class NumPyMultinomialLogistic:
    """A small but real softmax/multinomial logistic regression estimator.

    The fit method minimizes mean multiclass log-loss with L2 regularization.
    Label order follows np.unique(y), as in scikit-learn.
    """

    def __init__(self, iterations=400, learning_rate=0.08, l2=0.003):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2 = l2

    def __getitem__(self, index):
        # Backwards compatibility with tests that inspected sklearn's pipeline[-1].
        if index == -1:
            return self
        raise IndexError(index)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or X.shape[0] == 0:
            raise ValueError("Training X/y have incompatible dimensions")
        if not np.isfinite(X).all():
            raise ValueError("Training features contain NaN or infinite numbers")
        self.classes_, inverse = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Training requires at least two classes")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        xs = (X - self.mean_) / self.scale_
        n, d = xs.shape
        k = len(self.classes_)
        onehot = np.eye(k, dtype=float)[inverse]
        weights = np.zeros((d, k))
        bias = np.zeros(k)
        m_w = np.zeros_like(weights)
        v_w = np.zeros_like(weights)
        m_b = np.zeros_like(bias)
        v_b = np.zeros_like(bias)
        beta1, beta2 = 0.9, 0.999
        for it in range(1, self.iterations + 1):
            logits = xs @ weights + bias
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            error = (probabilities - onehot) / n
            grad_w = xs.T @ error + self.l2 * weights
            grad_b = error.sum(axis=0)
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            correction_w = (m_w / (1 - beta1 ** it)) / (np.sqrt(v_w / (1 - beta2 ** it)) + 1e-8)
            correction_b = (m_b / (1 - beta1 ** it)) / (np.sqrt(v_b / (1 - beta2 ** it)) + 1e-8)
            weights -= self.learning_rate * correction_w
            bias -= self.learning_rate * correction_b
        # Stored in sklearn-like (n_classes, n_features) format for existing tests.
        self.coef_ = weights.T.copy()
        self.intercept_ = bias.copy()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.coef_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Prediction features have wrong shape or nonfinite values")
        xs = (X - self.mean_) / self.scale_
        logits = xs @ self.coef_.T + self.intercept_
        logits -= np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / exps.sum(axis=1, keepdims=True)
