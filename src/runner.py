"""
Experiment runner for the ensemble-learning cyberattack classification study.

Usage (run the experiments one at a time, or together):

    python runner.py --experiment 1                       # sampling comparison
    python runner.py --experiment 2 --sampling bootstrap  # individual vs bagged
    python runner.py --experiment 2 --sampling subsampling
    python runner.py                                      # both (experiment 2 uses --sampling, default bootstrap)

Experiments
-----------
* Experiment 1 = sampling-strategy comparison (research question RQ2). Compares
  bootstrap vs subsampling for bagged KNN and bagged decision tree. After it
  finishes, the runner prints which strategy scored higher overall as a hint for
  choosing --sampling in experiment 2. The choice is always yours.
* Experiment 2 = individual vs bagged comparison (research question RQ1). Uses the
  single sampling strategy given by --sampling (default bootstrap) for BOTH
  bagged models. --sampling has no effect on experiment 1.

Responsibilities
----------------
1. main - parse flags, load data, split ONCE, fit the two
        preprocessing pipelines on training data only, run
        the selected experiment(s) and save CSV outputs.
2. run_rq1_ensemble_effect - individual models vs bagged ensembles with a chosen
                            sampling strategy (4 configurations, 3 PDFs).
3. run_rq2_sampling_strategy- bootstrap vs subsampling bagged ensembles
                                 (4 configurations, 3 PDFs).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are only written to PDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Makes sibling modules importable regardless of the terminal's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import BaggedDecisionTree, BaggedKNN, IndividualModels  
from pre_processing import CyberAttackPreprocessor  

plt.rcParams["pdf.fonttype"] = 42  

DATASET_NAME = "networkTraffic.csv"
NOT_APPLICABLE = "not_applicable"

METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro-precision",
    "macro_recall": "Macro-recall",
    "macro_f1": "Macro-F1",
}

# Model-configuration display names (also used as row labels in the heatmaps).
IND_KNN = "Individual KNN"
IND_TREE = "Individual decision tree"
BAG_KNN_BOOT = "Bagged KNN (bootstrap)"
BAG_KNN_SUB = "Bagged KNN (subsampling)"
BAG_TREE_BOOT = "Bagged decision tree (bootstrap)"
BAG_TREE_SUB = "Bagged decision tree (subsampling)"

# Maps display sampling names to the strings understood by models.py.
SAMPLING_METHOD = {"bootstrap": "bootstrap", "subsampling": "subsample"}


def bagged_name(kind: str, sampling: str) -> str:
    """Display name of a bagged configuration, e.g. 'Bagged KNN (subsampling)'."""
    algorithm = "KNN" if kind == "knn" else "decision tree"
    return f"Bagged {algorithm} ({sampling})"


@dataclass(frozen=True)
class ExperimentConfig:
    """Agreed, fixed experimental settings (nothing here is tuned)."""

    random_seed: int = 42
    test_fraction: float = 0.20
    sample_fraction: float = 0.8
    n_base_learners: int = 5
    n_neighbors: int = 20
    max_depth: int = 10


@dataclass
class ExperimentOutput:
    """What an experiment function returns."""

    metrics: pd.DataFrame        # one row per configuration (consolidated schema)
    comparisons: pd.DataFrame    # explicit metric-difference table
    predictions: pd.DataFrame    # long-format test predictions
    figure_paths: List[Path]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def find_project_root() -> Path:
    """
    Walks up from this file to the first folder holding data/networkTraffic.csv.

    Falls back to the parent of src/ if the dataset is absent,
    so a clear FileNotFoundError can be raised later with the expected location.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "data" / DATASET_NAME).is_file():
            return candidate
    return here.parent if here.name == "src" else here


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Accuracy and macro-averaged precision/recall/F1.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}; cannot evaluate.")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _run_configuration(name: str, fit_and_predict: Callable[[], np.ndarray]) -> np.ndarray:
    """
    Runs one configuration; on failure raise an error that names it.
    """
    start = time.perf_counter()
    try:
        y_pred = fit_and_predict()
    except Exception as exc: 
        raise RuntimeError(f"Configuration '{name}' FAILED: {type(exc).__name__}: {exc}") from exc
    print(f"    done: {name:<40s} ({time.perf_counter() - start:7.1f}s)")

    return np.asarray(y_pred)


def _check_ensemble(ensemble, kind: str, method: str, n_train: int, cfg: ExperimentConfig) -> None:
    """
    Verifies a fitted ensemble matches the agreed design and its sampling is valid.
    """
    assert len(ensemble.models_) == cfg.n_base_learners, "wrong number of base learners"
    assert len(ensemble.samples_) == cfg.n_base_learners, "wrong number of samples"
    for model in ensemble.models_:
        if kind == "knn":
            assert model.n_neighbors == cfg.n_neighbors, "base KNN has wrong k"
        else:
            assert model.max_depth == cfg.max_depth, "base tree has wrong max_depth"
    expected_size = math.ceil(round(cfg.sample_fraction * n_train, 9))
    any_duplicates = False
    for sample in ensemble.samples_:
        assert len(sample.indices) == expected_size, "sample size is not ceil(0.8 * n_train)"
        assert sample.indices.min() >= 0 and sample.indices.max() < n_train, \
            "sample references rows outside the training partition"
        unique = len(np.unique(sample.indices))
        if method == "subsample":
            assert unique == len(sample.indices), "subsample contains repeated rows"
        any_duplicates |= unique < len(sample.indices)
    if method == "bootstrap":
        assert any_duplicates, "bootstrap samples contain no repeated rows (replacement not used?)"


def _fit_predict_ensemble(
    kind: str, sampling: str, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Fits BaggedKNN / BaggedDecisionTree via the existing classes and predict test rows.
    """
    method = SAMPLING_METHOD[sampling]
    ensemble_cls = BaggedKNN if kind == "knn" else BaggedDecisionTree
    ensemble = ensemble_cls(
        sampling_method=method,
        sample_fraction=cfg.sample_fraction,
        random_seed=cfg.random_seed,
    )
    ensemble.fit(X_train, y_train)
    _check_ensemble(ensemble, kind, method, len(X_train), cfg)

    return ensemble.predict(X_test)


def _metrics_row(
    rq: str, name: str, algorithm: str, model_kind: str, sampling: str,
    metrics: Dict[str, float], cfg: ExperimentConfig, n_train: int, n_test: int,
) -> Dict[str, object]:
    """
    One row of the consolidated model_metrics table.
    """
    is_bagged = model_kind == "bagged"
    return {
        "research_question": rq,
        "model_name": name,
        "base_algorithm": algorithm,
        "model_type": model_kind,
        "sampling_strategy": sampling if is_bagged else NOT_APPLICABLE,
        "n_base_learners": cfg.n_base_learners if is_bagged else NOT_APPLICABLE,
        "sample_fraction": cfg.sample_fraction if is_bagged else NOT_APPLICABLE,
        "n_neighbors": cfg.n_neighbors if algorithm == "KNN" else NOT_APPLICABLE,
        "max_depth": cfg.max_depth if algorithm == "Decision tree" else NOT_APPLICABLE,
        "random_seed": cfg.random_seed,
        "test_fraction": cfg.test_fraction,
        "n_train": n_train,
        "n_test": n_test,
        **metrics,
    }


def _predictions_frame(rq: str, name: str, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Handles long-format predictions; the index is an identifier only, never a predictor.
    """
    return pd.DataFrame({
        "research_question": rq,
        "model_name": name,
        "test_row_index": y_test.index.to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": y_pred,
    })


def _difference_table(metrics: pd.DataFrame, pairs: List[Tuple[str, str, str]], interpretations: Tuple[str, str]) -> pd.DataFrame:
    """
    Builds metric differences comparison - reference (absolute, 0-1 scale).

    Args:
        metrics: Metrics table indexed by model_name.
        pairs: (algorithm, reference_model, comparison_model) triples.
        interpretations: Text used when the difference is positive / negative.
    """
    rows = []
    for algorithm, reference, comparison in pairs:
        for metric in METRIC_COLUMNS:
            ref_value = metrics.loc[reference, metric]
            cmp_value = metrics.loc[comparison, metric]
            diff = cmp_value - ref_value
            verdict = interpretations[0] if diff > 0 else interpretations[1] if diff < 0 else "no difference"
            rows.append({
                "algorithm": algorithm,
                "metric": METRIC_LABELS[metric],
                "reference_model": reference,
                "reference_value": ref_value,
                "comparison_model": comparison,
                "comparison_value": cmp_value,
                "difference": diff,
                "difference_definition": f"{comparison} - {reference}",
                "interpretation": verdict,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# -------------------------------- Plotting --------------------------------- #
# --------------------------------------------------------------------------- #
def _save_heatmap(table: pd.DataFrame, title: str, path: Path) -> None:
    """
    Annotated heatmap, fixed 0-1 colour scale. table: 4 models x 4 metrics.
    """
    assert table.shape == (4, 4), f"heatmap must be 4x4, got {table.shape}"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), labels=[METRIC_LABELS[m] for m in METRIC_COLUMNS], fontsize=10)
    ax.set_yticks(range(4), labels=list(table.index), fontsize=10)
    ax.set_xlabel("Evaluation metric", fontsize=11)
    ax.set_ylabel("Model configuration", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    for i in range(4):
        for j in range(4):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if value > 0.6 else "black")
    fig.colorbar(image, ax=ax, label="Metric value (0-1)")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _save_paired_bars(table: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    """
    Grouped bar chart: table has exactly 2 model rows x 4 metric columns.
    """
    assert table.shape == (2, 4), f"paired bar chart needs 2 models x 4 metrics, got {table.shape}"
    x = np.arange(4)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = ["#7f9fbf", "#c8553d"]
    for k, (model_name, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + (k - 0.5) * width, row.to_numpy(dtype=float), width,
                      label=model_name, color=colours[k])
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels=[METRIC_LABELS[m] for m in METRIC_COLUMNS], fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Evaluation metric", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=14)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 2 (research question RQ1): individual vs bagged
# --------------------------------------------------------------------------- #
def run_rq1_ensemble_effect(
    X_train_knn: pd.DataFrame, X_test_knn: pd.DataFrame,
    X_train_tree: pd.DataFrame, X_test_tree: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    figures_dir: Path, cfg: ExperimentConfig, sampling: str = "bootstrap",
) -> ExperimentOutput:
    """
    Trains/evaluates exactly four configurations on the shared split:
    Individual KNN, Bagged KNN, Individual decision tree and Bagged decision tree.
    Both bagged models use the same sampling strategy ("bootstrap" or
    "subsampling"). Individual models use the full training partition; ensembles
    sample internally. 
    
    Saves three PDF figures.
    """
    if sampling not in SAMPLING_METHOD:
        raise ValueError(f"sampling must be one of {list(SAMPLING_METHOD)}, got {sampling!r}.")
    rq = "RQ1"
    bag_knn, bag_tree = bagged_name("knn", sampling), bagged_name("tree", sampling)
    print(f"[Experiment 2 / RQ1] Individual vs bagged models (sampling: {sampling})")
    n_train, n_test = len(X_train_knn), len(X_test_knn)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def individual_knn() -> np.ndarray:
        model = IndividualModels(random_state=cfg.random_seed)
        model.train_knn(X_train_knn, y_train)
        assert model.knn_model.n_neighbors == cfg.n_neighbors
        return model.predict_knn(X_test_knn)

    def individual_tree() -> np.ndarray:
        model = IndividualModels(random_state=cfg.random_seed)
        model.train_decision_tree(X_train_tree, y_train)
        assert model.tree_model.max_depth == cfg.max_depth
        return model.predict_decision_tree(X_test_tree)

    configs = [
        (IND_KNN, "KNN", "individual", None, individual_knn),
        (bag_knn, "KNN", "bagged", sampling,
         lambda: _fit_predict_ensemble("knn", sampling, X_train_knn, y_train, X_test_knn, cfg)),
        (IND_TREE, "Decision tree", "individual", None, individual_tree),
        (bag_tree, "Decision tree", "bagged", sampling,
         lambda: _fit_predict_ensemble("tree", sampling, X_train_tree, y_train, X_test_tree, cfg)),
    ]

    metric_rows, prediction_frames = [], []
    for name, algorithm, kind, samp, fn in configs:
        y_pred = _run_configuration(name, fn)
        metric_rows.append(_metrics_row(rq, name, algorithm, kind, samp, compute_metrics(y_test, y_pred),
                                        cfg, n_train, n_test))
        prediction_frames.append(_predictions_frame(rq, name, y_test, y_pred))

    metrics = pd.DataFrame(metric_rows)
    by_model = metrics.set_index("model_name")[METRIC_COLUMNS]
    comparisons = _difference_table(
        by_model,
        [("KNN", IND_KNN, bag_knn), ("Decision tree", IND_TREE, bag_tree)],
        ("bagged model higher", "individual model higher"),
    )

    paths = [figures_dir / "rq1_heatmap.pdf",
             figures_dir / "rq1_knn_comparison.pdf",
             figures_dir / "rq1_decision_tree_comparison.pdf"]
    _save_heatmap(by_model.loc[[IND_KNN, bag_knn, IND_TREE, bag_tree]],
                  f"RQ1: Individual models vs bagged ensembles ({sampling} sampling)", paths[0])
    _save_paired_bars(by_model.loc[[IND_KNN, bag_knn]],
                      f"RQ1: Individual KNN vs bagged KNN ({sampling})", "Metric value", paths[1])
    _save_paired_bars(by_model.loc[[IND_TREE, bag_tree]],
                      f"RQ1: Individual decision tree vs bagged decision tree ({sampling})", "Metric value", paths[2])
    return ExperimentOutput(metrics, comparisons, pd.concat(prediction_frames, ignore_index=True), paths)


# --------------------------------------------------------------------------- #
# Experiment 1 (research question RQ2): sampling strategies
# --------------------------------------------------------------------------- #
def run_rq2_sampling_strategy(
    X_train_knn: pd.DataFrame, X_test_knn: pd.DataFrame,
    X_train_tree: pd.DataFrame, X_test_tree: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    figures_dir: Path, cfg: ExperimentConfig,
) -> ExperimentOutput:
    """
    Trains/evaluates exactly four bagged configurations (KNN and decision tree,
    each with bootstrap and subsampling). Individual models are NOT part of this
    experiment. Only the sampling strategy differs within each algorithm.
    """
    rq = "RQ2"
    print("[Experiment 1 / RQ2] Bootstrap vs subsampling")
    n_train, n_test = len(X_train_knn), len(X_test_knn)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def ensemble_fn(kind: str, sampling: str) -> Callable[[], np.ndarray]:
        X_tr, X_te = (X_train_knn, X_test_knn) if kind == "knn" else (X_train_tree, X_test_tree)
        return lambda: _fit_predict_ensemble(kind, sampling, X_tr, y_train, X_te, cfg)

    configs = [
        (BAG_KNN_BOOT, "KNN", "knn", "bootstrap"),
        (BAG_KNN_SUB, "KNN", "knn", "subsampling"),
        (BAG_TREE_BOOT, "Decision tree", "tree", "bootstrap"),
        (BAG_TREE_SUB, "Decision tree", "tree", "subsampling"),
    ]

    metric_rows, prediction_frames = [], []
    for name, algorithm, kind, sampling in configs:
        y_pred = _run_configuration(name, ensemble_fn(kind, sampling))
        metric_rows.append(_metrics_row(rq, name, algorithm, "bagged", sampling, compute_metrics(y_test, y_pred),
                                        cfg, n_train, n_test))
        prediction_frames.append(_predictions_frame(rq, name, y_test, y_pred))

    metrics = pd.DataFrame(metric_rows)
    by_model = metrics.set_index("model_name")[METRIC_COLUMNS]
    assert not any(n.startswith("Individual") for n in by_model.index), "individual model leaked into RQ2"
    comparisons = _difference_table(
        by_model,
        [("KNN", BAG_KNN_SUB, BAG_KNN_BOOT), ("Decision tree", BAG_TREE_SUB, BAG_TREE_BOOT)],
        ("favours bootstrap", "favours subsampling"),
    )

    paths = [figures_dir / "rq2_heatmap.pdf",
             figures_dir / "rq2_knn_sampling_comparison.pdf",
             figures_dir / "rq2_decision_tree_sampling_comparison.pdf"]
    _save_heatmap(by_model.loc[[BAG_KNN_BOOT, BAG_KNN_SUB, BAG_TREE_BOOT, BAG_TREE_SUB]],
                  "RQ2: Bootstrap vs subsampling in bagged ensembles", paths[0])
    _save_paired_bars(by_model.loc[[BAG_KNN_BOOT, BAG_KNN_SUB]],
                      "RQ2: Bagged KNN - bootstrap vs subsampling", "Metric value", paths[1])
    _save_paired_bars(by_model.loc[[BAG_TREE_BOOT, BAG_TREE_SUB]],
                      "RQ2: Bagged decision tree - bootstrap vs subsampling", "Metric value", paths[2])
    return ExperimentOutput(metrics, comparisons, pd.concat(prediction_frames, ignore_index=True), paths)


# --------------------------------------------------------------------------- #
# Data preparation and orchestration
# --------------------------------------------------------------------------- #
def load_dataset(path: Path, target: str) -> pd.DataFrame:
    """
    Loads the CSV and run basic sanity checks.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected data/{DATASET_NAME} under the project root, "
            "or pass --data /path/to/file.csv."
        )
    dataset = pd.read_csv(path)
    if target not in dataset.columns:
        raise ValueError(f"Target column '{target}' not found in {path.name}. Columns: {list(dataset.columns)}")
    if len(dataset) == 0:
        raise ValueError(f"{path.name} contains no observations.")
    if dataset[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values; cannot stratify or evaluate.")
    if "label" in dataset.columns:
        print("WARNING: the dataset contains a 'label' column.")
        
    return dataset


def _check_prepared_data(X_train, X_test, y_train, y_test, X_train_knn, X_test_knn, X_train_tree, X_test_tree) -> None:
    """
    Integrity checks on the single split and on the transformed matrices.
    """
    target = CyberAttackPreprocessor.TARGET
    assert target not in X_train.columns and target not in X_test.columns, "target found among predictors"
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "X/y length mismatch"
    assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index), "X/y rows misaligned"
    assert X_train.index.is_unique and X_test.index.is_unique, "duplicate row identifiers"
    assert len(X_train.index.intersection(X_test.index)) == 0, "train and test partitions overlap"
    for name, tr, te in (("knn", X_train_knn, X_test_knn), ("tree", X_train_tree, X_test_tree)):
        assert tr.index.equals(y_train.index) and te.index.equals(y_test.index), f"{name}: rows misaligned with labels"
        assert list(tr.columns) == list(te.columns), f"{name}: train/test feature columns differ"
        assert not tr.isna().any().any() and not te.isna().any().any(), \
            f"{name}: NaN values remain after preprocessing (KNN/tree cannot handle them)"
        assert target not in tr.columns, f"{name}: target present in transformed features"


def _check_cross_rq_reproducibility(rq1: ExperimentOutput, rq2: ExperimentOutput, sampling: str) -> None:
    """
    When both experiments ran together, the shared bagged models must match exactly.
    """
    for kind in ("knn", "tree"):
        name = bagged_name(kind, sampling)
        a = rq1.predictions.query("model_name == @name")["predicted_label"].to_numpy()
        b = rq2.predictions.query("model_name == @name")["predicted_label"].to_numpy()
        assert np.array_equal(a, b), f"'{name}' gave different predictions in both experiments despite fixed seed"


def _report_stronger_sampling(rq2: ExperimentOutput) -> None:
    """
    Prints which sampling strategy scored higher overall (a hint only, not a decision).

    'Overall' = mean of the four metrics, averaged over bagged KNN and bagged tree.
    """
    scores = (rq2.metrics.assign(mean_metric=rq2.metrics[METRIC_COLUMNS].mean(axis=1))
              .groupby("sampling_strategy")["mean_metric"].mean())
    best = scores.idxmax()
    print("\nOverall mean of the 4 metrics across both bagged algorithms:")
    for strategy, value in scores.items():
        print(f"    {strategy:<12s} {value:.6f}")
    if np.isclose(scores.max(), scores.min(), rtol=0, atol=1e-12):
        print("  The strategies are tied overall; choose --sampling based on your own criterion.")
    else:
        print(f"  Higher overall: {best}.  To use it in experiment 2:  "
              f"python run_experiments.py --experiment 2 --sampling {best}")
    print("  (Check rq2_comparisons.csv per algorithm and metric before deciding.)")


def _update_results_csv(path: Path, new_rows: pd.DataFrame, research_question: str) -> None:
    """
    Writes new_rows into a shared results CSV, replacing only this RQ's old rows.
    """
    if path.is_file():
        existing = pd.read_csv(path, keep_default_na=False)
        existing = existing[existing["research_question"] != research_question]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows = new_rows.sort_values("research_question", kind="stable")
    new_rows.to_csv(path, index=False)


def _save_experiment(output: ExperimentOutput, rq: str, results_dir: Path) -> None:
    """
    Persists one experiment's tables.
    """
    output.comparisons.to_csv(results_dir / f"{rq.lower()}_comparisons.csv", index=False)
    _update_results_csv(results_dir / "model_metrics.csv", output.metrics, rq)
    _update_results_csv(results_dir / "predictions.csv", output.predictions, rq)


def _print_summary(output: ExperimentOutput) -> None:
    summary = output.metrics[["research_question", "model_name", *METRIC_COLUMNS]]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: List[str] | None = None) -> None:
    """
    Orchestrates the selected experiment(s).
    """
    parser = argparse.ArgumentParser(
        description="Run the ensemble experiments. Experiment 1 = bootstrap vs subsampling (RQ2); "
                    "experiment 2 = individual vs bagged models (RQ1).")
    parser.add_argument("--experiment", choices=["1", "2", "all"], default="all",
                        help="1 = sampling comparison, 2 = individual vs bagged, all = both (default).")
    parser.add_argument("--sampling", choices=list(SAMPLING_METHOD), default=None,
                        help="Sampling strategy for the bagged models in experiment 2 "
                             "(default: bootstrap). Ignored by experiment 1.")
    parser.add_argument("--data", type=Path, default=None, help=f"Path to {DATASET_NAME} (default: <root>/data/).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <root>/output).")
    args = parser.parse_args(argv)

    run_exp1 = args.experiment in ("1", "all")
    run_exp2 = args.experiment in ("2", "all")
    sampling = args.sampling or "bootstrap"
    if args.experiment == "1" and args.sampling is not None:
        print("Note: --sampling is ignored by experiment 1 (it always compares both strategies).")
    if run_exp2:
        origin = "from --sampling" if args.sampling else "default"
        print(f"Experiment 2 will use '{sampling}' sampling for the bagged models ({origin}).")

    cfg = ExperimentConfig()
    root = find_project_root()
    data_path = args.data or root / "data" / DATASET_NAME
    output_dir = args.output_dir or root / "output"
    results_dir, figures_dir = output_dir / "results", output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Loads data, instantiate the existing preprocessor.
    dataset = load_dataset(data_path, CyberAttackPreprocessor.TARGET)
    print(f"Loaded {data_path} : {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    preprocessor = CyberAttackPreprocessor(dataset)

    # ONE stratified 80/20 split shared by every configuration.
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        test_size=cfg.test_fraction, random_state=cfg.random_seed
    )
    print(f"Split once: {len(X_train)} train / {len(X_test)} test (stratified on attack_cat, seed {cfg.random_seed})")

    # Fits each pipeline on TRAINING data only, then transform the test data.
    X_train_knn = preprocessor.preprocess_data(X_train, model_type="knn")
    X_test_knn = preprocessor.transform_test(X_test, model_type="knn")
    X_train_tree = preprocessor.preprocess_data(X_train, model_type="tree")
    X_test_tree = preprocessor.transform_test(X_test, model_type="tree")
    _check_prepared_data(X_train, X_test, y_train, y_test, X_train_knn, X_test_knn, X_train_tree, X_test_tree)
    print(f"Features after preprocessing: KNN {X_train_knn.shape[1]}, tree {X_train_tree.shape[1]}")

    shared = dict(X_train_knn=X_train_knn, X_test_knn=X_test_knn, X_train_tree=X_train_tree,
                  X_test_tree=X_test_tree, y_train=y_train, y_test=y_test, figures_dir=figures_dir, cfg=cfg)

    rq2_results = rq1_results = None
    if run_exp1:
        rq2_results = run_rq2_sampling_strategy(**shared)
        _save_experiment(rq2_results, "RQ2", results_dir)
        print("\nExperiment 1 results:")
        _print_summary(rq2_results)
        _report_stronger_sampling(rq2_results)
    if run_exp2:
        rq1_results = run_rq1_ensemble_effect(**shared, sampling=sampling)
        _save_experiment(rq1_results, "RQ1", results_dir)
        print("\nExperiment 2 results:")
        _print_summary(rq1_results)
    if rq1_results is not None and rq2_results is not None:
        _check_cross_rq_reproducibility(rq1_results, rq2_results, sampling)

    print("\nResults directory:", results_dir)
    print("Figures directory:", figures_dir)


if __name__ == "__main__":
    main()