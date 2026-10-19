# Ensemble Learning for Multiclass Cyberattack Classification

This assignment studies how bagging affects KNN and decision tree classifiers on network traffic data, with `attack_cat` as the target. It has two experiments:

1. **Sampling strategy:** bootstrap sampling vs subsampling without replacement for bagged KNN and bagged decision trees.
2. **Individual vs bagged:** each individual classifier compared with its bagged ensemble, using a chosen sampling strategy.

## Project structure

```
data/           networkTraffic.csv (not tracked in git)
report/         report files
src/
  preprocessing.py     CyberAttackPreprocessor
  models.py            SamplingStrategies, IndividualModels, BaggedKNN, BaggedDecisionTree
  run_experiments.py   experiment runner
output/
  results/             CSV results
  figures/             PDF figures
```

## Setup

Requires Python 3 with `numpy`, `pandas`, `scikit-learn` and `matplotlib`:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Place the dataset at `data/networkTraffic.csv`.

## Running the experiments

From the project root:

```bash
cd src

# Experiment 1: bootstrap vs subsampling
python run_experiments.py --experiment 1

# Experiment 2: individual vs bagged models (default sampling: bootstrap)
python run_experiments.py --experiment 2
python run_experiments.py --experiment 2 --sampling subsampling

# Both experiments in sequence
python run_experiments.py
```