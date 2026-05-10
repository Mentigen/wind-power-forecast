"""Optimal ensemble weight finder using scipy.optimize.minimize.

Finds weights w_i for N models such that the weighted average of
their OOF (out-of-fold) predictions minimizes the hackathon metric:

    score = mean(|actual - sum(w_i * pred_i)|) / 90.09 * 100

subject to: sum(w_i) = 1, 0 <= w_i <= 1 for all i.
"""
import numpy as np
from scipy.optimize import minimize
from config import INSTALLED_CAPACITY


HACKATHON_CAPACITY = INSTALLED_CAPACITY  # 90.09 MW


def hackathon_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MAE matching the hackathon scoring formula."""
    return float(np.mean(np.abs(y_true - y_pred)) / HACKATHON_CAPACITY * 100.0)


class EnsembleOptimizer:
    """Find optimal blending weights for an ensemble of regression models.

    Usage
    -----
    opt = EnsembleOptimizer()
    opt.fit(oof_predictions, y_true)   # oof_predictions: (n_samples, n_models)
    final_pred = opt.predict(test_predictions)   # test_predictions: (n_samples, n_models)
    """

    def __init__(self, n_restarts: int = 20, seed: int = 42):
        self.n_restarts = n_restarts
        self.seed = seed
        self.weights_ = None
        self.best_score_ = None

    def _objective(self, weights: np.ndarray, preds: np.ndarray, y: np.ndarray) -> float:
        blended = preds @ weights
        blended = np.clip(blended, 0.0, HACKATHON_CAPACITY)
        return hackathon_metric(y, blended)

    def fit(self, oof_predictions: np.ndarray, y_true: np.ndarray) -> "EnsembleOptimizer":
        """Find optimal weights by minimizing hackathon metric on OOF predictions.

        Parameters
        ----------
        oof_predictions : array of shape (n_samples, n_models)
        y_true : array of shape (n_samples,)
        """
        n_models = oof_predictions.shape[1]
        rng = np.random.RandomState(self.seed)

        best_weights = np.ones(n_models) / n_models
        best_score = self._objective(best_weights, oof_predictions, y_true)

        # Equality constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models

        for i in range(self.n_restarts):
            # Random Dirichlet starting point
            w0 = rng.dirichlet(np.ones(n_models))
            result = minimize(
                self._objective,
                w0,
                args=(oof_predictions, y_true),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_weights = result.x

        self.weights_ = best_weights / best_weights.sum()  # renormalize for safety
        self.best_score_ = best_score
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Apply optimal weights to test predictions.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_models)

        Returns
        -------
        blended : array of shape (n_samples,)
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        blended = predictions @ self.weights_
        return np.clip(blended, 0.0, HACKATHON_CAPACITY)

    def report(self) -> str:
        if self.weights_ is None:
            return "Not fitted yet."
        lines = [f"Optimal weights: {np.round(self.weights_, 4).tolist()}",
                 f"OOF hackathon score: {self.best_score_:.4f}%"]
        return "\n".join(lines)
