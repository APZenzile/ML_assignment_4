"""
Model implementations for the multiclass cyberattack-classification assignment.

This single module contains the four required components:

1. SamplingStrategies - bootstrap sampling and subsampling without replacement.
2. IndividualModels   - the baseline KNN (k=20) and decision tree (max_depth=10).
3. BaggedKNN          - five-learner homogeneous KNN ensemble with majority voting.
4. BaggedDecisionTree - five-learner homogeneous decision-tree ensemble with
                            majority voting.

Majority-voting tie-break rule
------------------------------
Each learner casts one vote per test observation. The class with the most votes
wins. If several classes tie, the winner is the tied class that comes FIRST in the
ensemble's class ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ArrayLike = Union[pd.DataFrame, np.ndarray]
LabelLike = Union[pd.Series, np.ndarray, Sequence[Any]]

# Fixed experimental settings.
DEFAULT_N_LEARNERS = 5
DEFAULT_N_NEIGHBORS = 20
DEFAULT_MAX_DEPTH = 10
DEFAULT_SAMPLE_FRACTION = 0.8
DEFAULT_RANDOM_SEED = 42

BOOTSTRAP = "bootstrap"
SUBSAMPLE = "subsample"
VALID_SAMPLING_METHODS = (BOOTSTRAP, SUBSAMPLE)


# --------------------------------------------------------------------------- #
# ----------------- Shared validation helpers ------------------------------- #
# --------------------------------------------------------------------------- #
def _validate_xy(X: ArrayLike, y: LabelLike, context: str) -> np.ndarray:
    """
    Validates a feature matrix / label vector pair and return y as a 1-D array.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"{context}: X must be a pandas DataFrame or numpy ndarray, "
            f"got {type(X).__name__}."
        )
    if X.ndim != 2:
        raise ValueError(f"{context}: X must be 2-dimensional, got {X.ndim} dimension(s).")
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"{context}: y must be 1-dimensional, got shape {y_arr.shape}.")
    if X.shape[0] == 0:
        raise ValueError(f"{context}: training data is empty.")
    if X.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"{context}: X has {X.shape[0]} rows but y has {y_arr.shape[0]} labels; "
            "they must be the same length."
        )
    return y_arr


def _validate_features_for_prediction(X: ArrayLike, context: str) -> None:
    """
    Validates a feature matrix passed to predict.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"{context}: X must be a pandas DataFrame or numpy ndarray, "
            f"got {type(X).__name__}."
        )
    if X.ndim != 2:
        raise ValueError(f"{context}: X must be 2-dimensional, got {X.ndim} dimension(s).")
    if X.shape[0] == 0:
        raise ValueError(f"{context}: X contains no rows to predict.")


def _take_rows(X: ArrayLike, indices: np.ndarray) -> ArrayLike:
    """
    Positional row selection that returns a copy.
    """
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices].copy()
    return X[indices].copy()


# --------------------------------------------------------------------------- #
# 1. SamplingStrategies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    """
    One paired feature/label sample drawn from a training partition.

    Attributes:
        X: Sampled feature rows.
        y: Sampled labels as a 1-D numpy array, aligned row-for-row with X.
        indices: Positional row indices (into the original training partition)
            that were selected, in draw order. Repeats are possible for
            bootstrap sampling and impossible for subsampling.
    """

    X: ArrayLike
    y: np.ndarray
    indices: np.ndarray


class SamplingStrategies:
    """
    Generates independent paired (X, y) training samples for the ensembles.

    Both strategies draw the same number of rows per sample,
    ceil(sample_fraction * n_train):

    Args:
        n_samples: Number of samples to generate per call (default 5).
        sample_fraction: Fraction of the training rows in each sample, in (0, 1]
            (default 0.8).
        random_seed: Non-negative integer seed (default 42).
    """

    def __init__(
        self,
        n_samples: int = DEFAULT_N_LEARNERS,
        sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        if isinstance(sample_fraction, bool) or not isinstance(sample_fraction, (int, float, np.floating)):
            raise TypeError(f"sample_fraction must be a number, got {type(sample_fraction).__name__}.")
        if not (0.0 < float(sample_fraction) <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction!r}.")
        if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
            raise ValueError(f"random_seed must be a non-negative integer, got {random_seed!r}.")
        self.n_samples = int(n_samples)
        self.sample_fraction = float(sample_fraction)
        self.random_seed = int(random_seed)

    # ----------------------- public API --------------------------------------------- #
    def sample_size(self, n_rows: int) -> int:
        """
        Returns the number of rows per sample: ceil(sample_fraction * n_rows).

        The product is rounded to 9 decimals before ceil so floating-point
        noise (e.g. 0.7 * 10 == 7.000000000000001) does not inflate the size.
        The result is at least 1 and never exceeds n_rows.
        """
        if n_rows < 1:
            raise ValueError(f"n_rows must be at least 1, got {n_rows}.")
        size = math.ceil(round(self.sample_fraction * n_rows, 9))
        return int(min(max(size, 1), n_rows))

    def bootstrap_samples(self, X_train: ArrayLike, y_train: LabelLike) -> List[Sample]:
        """
        Draws n_samples bootstrap samples (with replacement).
        """
        return self._generate(X_train, y_train, replace=True)

    def subsamples(self, X_train: ArrayLike, y_train: LabelLike) -> List[Sample]:
        """
        Draws n_samples subsamples (without replacement within each sample).
        """
        return self._generate(X_train, y_train, replace=False)

    def generate_samples(
        self, X_train: ArrayLike, y_train: LabelLike, method: str = BOOTSTRAP
    ) -> List[Sample]:
        """
        Draws n_samples samples using the named method.

        Args:
            X_train: Training features (DataFrame or ndarray).
            y_train: Training labels, same length as X_train.
            method: "bootstrap" or "subsample".

        Returns:
            A list of n_samples Sample objects (X, y, indices).
        """
        if method == BOOTSTRAP:
            return self.bootstrap_samples(X_train, y_train)
        if method == SUBSAMPLE:
            return self.subsamples(X_train, y_train)
        raise ValueError(f"Unknown sampling method {method!r}; expected one of {VALID_SAMPLING_METHODS}.")

    # -- internals --------------------------------------------------------- #
    def _generate(self, X_train: ArrayLike, y_train: LabelLike, replace: bool) -> List[Sample]:
        y_arr = _validate_xy(X_train, y_train, "SamplingStrategies")
        n_rows = X_train.shape[0]
        size = self.sample_size(n_rows)
        child_seeds = np.random.SeedSequence(self.random_seed).spawn(self.n_samples)

        samples: List[Sample] = []
        for seed in child_seeds:
            rng = np.random.default_rng(seed)
            if replace:
                idx = rng.integers(0, n_rows, size=size)
            else:
                idx = rng.choice(n_rows, size=size, replace=False)
            samples.append(
                Sample(X=_take_rows(X_train, idx), y=y_arr[idx].copy(), indices=idx)
            )
        return samples


# --------------------------------------------------------------------------- #
# 2. IndividualModels
# --------------------------------------------------------------------------- #
class IndividualModels:
    """
    Baseline individual classifiers trained on the FULL training partition.

    * KNN: n_neighbors=20 - give it the standardised features.
    * Decision tree: max_depth=10 with a fixed random_state

    Args:
        random_state: Fixed random_state for the decision tree (default 42).
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_SEED) -> None:
        self.n_neighbors = DEFAULT_N_NEIGHBORS
        self.max_depth = DEFAULT_MAX_DEPTH
        self.random_state = random_state
        self.knn_model: KNeighborsClassifier | None = None
        self.tree_model: DecisionTreeClassifier | None = None

    def train_knn(self, X_train: ArrayLike, y_train: LabelLike) -> "IndividualModels":
        """Fits the individual KNN (k=20) on standardised training data."""
        y_arr = _validate_xy(X_train, y_train, "IndividualModels.train_knn")
        model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        model.fit(X_train, y_arr)
        self.knn_model = model

        return self

    def predict_knn(self, X_test: ArrayLike) -> np.ndarray:
        """Predicts with the individual KNN; returns a 1-D array of class labels."""
        if self.knn_model is None:
            raise RuntimeError("Individual KNN has not been trained. Call train_knn() first.")
        _validate_features_for_prediction(X_test, "IndividualModels.predict_knn")
        return np.asarray(self.knn_model.predict(X_test))

    def train_decision_tree(self, X_train: ArrayLike, y_train: LabelLike) -> "IndividualModels":
        """Fits the individual decision tree (max_depth=10) on unscaled training data."""
        y_arr = _validate_xy(X_train, y_train, "IndividualModels.train_decision_tree")
        model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        model.fit(X_train, y_arr)
        self.tree_model = model

        return self

    def predict_decision_tree(self, X_test: ArrayLike) -> np.ndarray:
        """Predicts with the individual decision tree; returns a 1-D array of labels."""
        if self.tree_model is None:
            raise RuntimeError("Individual decision tree has not been trained. Call train_decision_tree() first.")
        _validate_features_for_prediction(X_test, "IndividualModels.predict_decision_tree")
        return np.asarray(self.tree_model.predict(X_test))


# --------------------------------------------------------------------------- #
# Shared majority-voting helper
# --------------------------------------------------------------------------- #
def majority_vote(predictions: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    """
    Combines learner predictions by majority vote with deterministic tie-breaking.

    Args:
        predictions: Array of shape (n_learners, n_observations) holding the
            predicted class label from each learner.
        class_order: 1-D array of all classes in the ensemble's canonical order
            (sorted unique training labels). Earlier classes win ties.

    Returns:
        1-D array of length n_observations with the winning label per
        observation. A tie between classes is resolved in favour of the tied class
        that appears first in class_order (argmax returns the first maximum).
    """
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape (n_learners, n_observations).")
    class_order = np.asarray(class_order)

    # Maps each predicted label to its position in the canonical class order.
    positions = np.searchsorted(class_order, predictions)
    positions = np.clip(positions, 0, len(class_order) - 1)
    if not np.array_equal(class_order[positions], predictions):
        raise ValueError("A learner predicted a label that is not in class_order.")

    n_obs = predictions.shape[1]
    votes = np.zeros((len(class_order), n_obs), dtype=np.int64)
    for learner_positions in positions:  # one row of positions per learner
        votes[learner_positions, np.arange(n_obs)] += 1
    return class_order[np.argmax(votes, axis=0)]


# --------------------------------------------------------------------------- #
# Shared ensemble machinery (private)
# --------------------------------------------------------------------------- #
class _BaggedEnsemble:
    """
    Common fit/predict logic for the two homogeneous bagging ensembles.
    """

    def __init__(
        self,
        sampling_method: str = BOOTSTRAP,
        sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        if sampling_method not in VALID_SAMPLING_METHODS:
            raise ValueError(
                f"Unknown sampling method {sampling_method!r}; "
                f"expected one of {VALID_SAMPLING_METHODS}."
            )
        self.n_learners = DEFAULT_N_LEARNERS
        self.sampling_method = sampling_method
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.models_: List[Any] = []
        self.classes_: np.ndarray | None = None
        self.samples_: List[Sample] = []

    def _make_learner(self) -> Any: 
        raise NotImplementedError

    def fit(self, X_train: ArrayLike, y_train: LabelLike, sampling_method: str | None = None) -> "_BaggedEnsemble":
        """
        Samples the training data and train one independent learner per sample.

        Args:
            X_train: Preprocessed training features.
            y_train: Training labels.
            sampling_method: Optional override of the constructor's method
                ("bootstrap" or "subsample").
        """
        y_arr = _validate_xy(X_train, y_train, f"{type(self).__name__}.fit")
        if sampling_method is not None:
            if sampling_method not in VALID_SAMPLING_METHODS:
                raise ValueError(
                    f"Unknown sampling method {sampling_method!r}; "
                    f"expected one of {VALID_SAMPLING_METHODS}."
                )
            self.sampling_method = sampling_method

        # sampling (each sample drawn from the full training partition).
        sampler = SamplingStrategies(
            n_samples=self.n_learners,
            sample_fraction=self.sample_fraction,
            random_seed=self.random_seed,
        )
        samples = sampler.generate_samples(X_train, y_arr, self.sampling_method)

        # independent training, one fresh learner per sample.
        models = []
        for sample in samples:
            learner = self._make_learner()
            learner.fit(sample.X, sample.y)
            models.append(learner)

        self.samples_ = samples
        self.models_ = models
        self.classes_ = np.unique(y_arr)

        return self

    def predict(self, X_test: ArrayLike) -> np.ndarray:
        """
        Predicts by majority vote over the five learners.

        Returns a 1-D array with one predicted class per test observation. Ties are
        broken by the sorted order of the training labels (see module docstring).
        """
        if not self.models_ or self.classes_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been trained. Call fit() before predict()."
            )
        _validate_features_for_prediction(X_test, f"{type(self).__name__}.predict")
        all_predictions = np.vstack([np.asarray(m.predict(X_test)) for m in self.models_])

        return majority_vote(all_predictions, self.classes_)


# --------------------------------------------------------------------------- #
# 3. BaggedKNN
# --------------------------------------------------------------------------- #
class BaggedKNN(_BaggedEnsemble):
    """
    Homogeneous bagging ensemble of five KNN classifiers (k=20 each).

    Args:
        sampling_method: "bootstrap" (default) or "subsample".
        sample_fraction: Fraction of training rows per sample (default 0.8).
        random_seed: Seed for the sampling (default 42).
    """

    def __init__(
        self,
        sampling_method: str = BOOTSTRAP,
        sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        super().__init__(sampling_method, sample_fraction, random_seed)
        self.n_neighbors = DEFAULT_N_NEIGHBORS

    def _make_learner(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)


# --------------------------------------------------------------------------- #
# 4. BaggedDecisionTree
# --------------------------------------------------------------------------- #
class BaggedDecisionTree(_BaggedEnsemble):
    """
    Homogeneous bagging ensemble of five decision trees (max_depth=10 each).

    Args:
        sampling_method: "bootstrap" (default) or "subsample".
        sample_fraction: Fraction of training rows per sample (default 0.8).
        random_seed: Seed for sampling and for every tree's random_state (default 42).
    """

    def __init__(
        self,
        sampling_method: str = BOOTSTRAP,
        sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        super().__init__(sampling_method, sample_fraction, random_seed)
        self.max_depth = DEFAULT_MAX_DEPTH

    def _make_learner(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_seed)