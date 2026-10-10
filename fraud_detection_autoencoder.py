"""
Unsupervised Credit Card Fraud Detection with a PyOD AutoEncoder.

Hands-On Assignment 4
---------------------
This script trains PyOD's deep-learning AutoEncoder on the anonymised credit
card transactions dataset from Kaggle (284,807 transactions, 492 frauds) and
uses the *reconstruction error* of each transaction as its anomaly score.

How it works
    1. The AutoEncoder compresses each transaction (29 features) into a small
       latent representation and then tries to rebuild the original input.
    2. Because > 99.8% of transactions are legitimate, the network learns to
       rebuild "normal" behaviour well.
    3. Fraudulent transactions look different, so they are rebuilt poorly and
       get a high reconstruction error -> high anomaly score -> flagged.

The model is trained WITHOUT labels (unsupervised). The `Class` labels are
only used afterwards to evaluate how well the anomaly scores catch fraud.

Usage
    python fraud_detection_autoencoder.py --data data/creditcard.csv
    python fraud_detection_autoencoder.py --data data/creditcard.csv --epochs 20

Author: <Your Name>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: save plots without a display.
import matplotlib.colors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyod.models.auto_encoder import AutoEncoder  # noqa: E402
from pyod.utils.utility import precision_n_scores  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Class"     # 1 = fraud, 0 = legitimate
AMOUNT_COLUMN = "Amount"    # transaction value, heavily right-skewed
DROP_COLUMNS = ["Time"]     # seconds since first transaction: no fraud signal
# V1..V28 are already PCA-transformed by the dataset provider.
RANDOM_SEED = 42

logger = logging.getLogger("fraud_detection")


@dataclass
class ExperimentConfig:
    """All tunable settings for one experiment, kept in one place."""

    data_path: Path
    output_dir: Path = Path("outputs")
    test_size: float = 0.30
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_neurons: list[int] = field(default_factory=lambda: [24, 12, 24])
    dropout_rate: float = 0.2
    random_state: int = RANDOM_SEED


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def configure_logging(log_file: Path) -> None:
    """Log to both the console and a file so the run can be reviewed later."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                  datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def set_seeds(seed: int) -> None:
    """Fix random seeds so results are reproducible between runs."""
    np.random.seed(seed)
    try:
        import torch  # PyOD's AutoEncoder is built on PyTorch.

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a PyOD dependency
        pass


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> pd.DataFrame:
    """Read the Kaggle CSV and check that it has the expected columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/whenamancodes/fraud-detection "
            "and place it in the data/ folder."
        )

    df = pd.read_csv(path)
    missing = {TARGET_COLUMN, AMOUNT_COLUMN, *DROP_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Log a short exploratory summary of the data."""
    counts = df[TARGET_COLUMN].value_counts()
    fraud_ratio = counts.get(1, 0) / len(df)
    logger.info("Dataset shape: %d rows x %d columns", *df.shape)
    logger.info("Missing values: %d", int(df.isna().sum().sum()))
    logger.info("Legitimate transactions: %d", counts.get(0, 0))
    logger.info("Fraudulent transactions: %d (%.4f%% of all data)",
                counts.get(1, 0), fraud_ratio * 100)


def split_and_scale(df: pd.DataFrame, cfg: ExperimentConfig):
    """Prepare features, create a stratified train/test split, scale Amount.

    * `Time` is dropped: it only counts seconds from the first transaction.
    * `Amount` ranges from 0 to ~25,000. Without a log transform a handful of
      very large (but legitimate) purchases produce enormous reconstruction
      errors and crowd real fraud out of the top of the ranking.
    * The scaler is fitted on the training set only, to avoid leaking
      information from the test set into training.
    """
    features = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    features[AMOUNT_COLUMN] = np.log1p(features[AMOUNT_COLUMN])
    labels = df[TARGET_COLUMN].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=cfg.test_size,
        stratify=labels,              # keep the same fraud ratio in each split
        random_state=cfg.random_state,
    )

    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[[AMOUNT_COLUMN]] = scaler.fit_transform(x_train[[AMOUNT_COLUMN]])
    x_test[[AMOUNT_COLUMN]] = scaler.transform(x_test[[AMOUNT_COLUMN]])

    logger.info("Train set: %d rows (%d frauds)", len(y_train), y_train.sum())
    logger.info("Test set : %d rows (%d frauds)", len(y_test), y_test.sum())
    return (x_train.to_numpy(dtype=np.float32), x_test.to_numpy(dtype=np.float32),
            y_train, y_test)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TrackedAutoEncoder(AutoEncoder):
    """PyOD AutoEncoder that also records the mean training loss per epoch.

    PyOD does not keep a loss history, so we hook into two of its extension
    points: `training_forward` (called once per mini-batch) and `epoch_update`
    (called at the end of every epoch). Training behaviour is unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_history_: list[float] = []
        self._batch_losses: list[float] = []

    def training_forward(self, batch_data):
        loss = super().training_forward(batch_data)
        self._batch_losses.append(loss)
        return loss

    def epoch_update(self):
        super().epoch_update()
        epoch_loss = float(np.mean(self._batch_losses))
        self.loss_history_.append(epoch_loss)
        self._batch_losses = []
        logger.info("  Epoch %2d/%d - reconstruction loss: %.4f",
                    len(self.loss_history_), self.epoch_num, epoch_loss)


def build_autoencoder(cfg: ExperimentConfig, contamination: float) -> AutoEncoder:
    """Create the PyOD AutoEncoder.

    `contamination` is the expected share of outliers. PyOD uses it to pick the
    reconstruction-error threshold that turns scores into 0/1 predictions.
    """
    return TrackedAutoEncoder(
        hidden_neuron_list=cfg.hidden_neurons,  # 29 -> 24 -> 12 -> 24 -> 29
        hidden_activation_name="relu",
        epoch_num=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        dropout_rate=cfg.dropout_rate,
        batch_norm=True,
        contamination=contamination,
        preprocessing=True,  # PyOD standardises the inputs internally
        random_state=cfg.random_state,
        verbose=0,  # per-epoch progress is logged by TrackedAutoEncoder
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute fraud-detection metrics suited to a heavily imbalanced dataset.

    Accuracy is misleading here (predicting 'never fraud' is 99.8% accurate),
    so we focus on ROC-AUC, PR-AUC, precision@n, recall and the confusion
    matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc_average_precision": round(
            float(average_precision_score(y_true, scores)), 4),
        "precision_at_n": round(float(precision_n_scores(y_true, scores)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "fraud_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "fraud_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
    }


def recall_at_top_k(y_true: np.ndarray, scores: np.ndarray,
                    fractions=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Share of all frauds caught if analysts review only the top-k% riskiest
    transactions. This is how such a model is used in practice."""
    order = np.argsort(scores)[::-1]       # highest anomaly score first
    total_frauds = y_true.sum()
    result = {}
    for frac in fractions:
        k = int(len(scores) * frac)
        caught = y_true[order[:k]].sum()
        result[f"top_{frac:.1%}"] = {"reviewed": k, "frauds_caught": int(caught),
                                     "recall": round(caught / total_frauds, 4)}
    return result


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_plots(y_true, scores, threshold, history, metrics, out_dir: Path) -> None:
    """Save a 2x2 figure: training loss, score distribution, ROC and PR curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Training loss per epoch (reconstruction error on training data).
    ax = axes[0, 0]
    if history:
        ax.plot(range(1, len(history) + 1), history, marker="o", color="#2a6fdb")
    ax.set(title="AutoEncoder training loss", xlabel="Epoch",
           ylabel="Mean reconstruction loss (MSE)")
    ax.grid(alpha=0.3)

    # 2. Distribution of anomaly scores (log scale shows both classes).
    ax = axes[0, 1]
    bins = np.logspace(np.log10(scores.min() + 1e-6), np.log10(scores.max()), 60)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="Legitimate",
            color="#2a6fdb", density=True)
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.7, label="Fraud",
            color="#d6452c", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label="Threshold")
    ax.set(xscale="log", title="Reconstruction-error scores (test set)",
           xlabel="Anomaly score (log scale)", ylabel="Density")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend()

    # 3. ROC curve.
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax.plot(fpr, tpr, color="#2a6fdb", label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="Random")
    ax.set(title="ROC curve", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # 4. Precision-Recall curve (more informative for rare fraud).
    ax = axes[1, 1]
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ax.plot(recall, precision, color="#d6452c",
            label=f"AP = {metrics['pr_auc_average_precision']:.3f}")
    ax.axhline(y_true.mean(), linestyle=":", color="grey",
               label=f"Baseline = {y_true.mean():.4f}")
    ax.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("PyOD AutoEncoder - Credit Card Fraud Detection", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=150)
    plt.close(fig)

    # Separate confusion-matrix figure.
    cm = np.array([[metrics["true_negatives"], metrics["false_positives"]],
                   [metrics["false_negatives"], metrics["true_positives"]]])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=13,
                color="white" if value > cm.max() / 10 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"],
           xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(cfg: ExperimentConfig) -> dict:
    """Run the full experiment: load -> split -> train -> score -> evaluate."""
    set_seeds(cfg.random_state)

    logger.info("=" * 64)
    logger.info("STEP 1: Load and explore the data")
    df = load_dataset(cfg.data_path)
    describe_dataset(df)

    logger.info("=" * 64)
    logger.info("STEP 2: Stratified train/test split and feature scaling")
    x_train, x_test, y_train, y_test = split_and_scale(df, cfg)

    # Expected outlier share = fraud rate. Only this summary statistic is used
    # for training; individual labels are never shown to the model.
    contamination = float(y_train.mean())
    logger.info("Contamination (expected fraud rate): %.5f", contamination)

    logger.info("=" * 64)
    logger.info("STEP 3: Train the AutoEncoder (unsupervised - no labels)")
    model = build_autoencoder(cfg, contamination)
    logger.info("Architecture: %d -> %s -> %d", x_train.shape[1],
                " -> ".join(map(str, cfg.hidden_neurons)), x_train.shape[1])
    start = time.perf_counter()
    model.fit(x_train)  # NOTE: y_train is deliberately NOT passed.
    logger.info("Training finished in %.1f seconds", time.perf_counter() - start)

    logger.info("=" * 64)
    logger.info("STEP 4: Score unseen test transactions")
    test_scores = model.decision_function(x_test)  # reconstruction errors
    test_pred = model.predict(x_test)              # 1 if score > threshold
    logger.info("Decision threshold (reconstruction error): %.4f",
                model.threshold_)
    logger.info("Transactions flagged as fraud: %d of %d",
                int(test_pred.sum()), len(test_pred))

    logger.info("=" * 64)
    logger.info("STEP 5: Evaluate against the true labels")
    metrics = evaluate(y_test, test_scores, test_pred)
    for name, value in metrics.items():
        logger.info("  %-26s %s", name, value)

    logger.info("Classification report:\n%s", classification_report(
        y_test, test_pred, target_names=["Legitimate", "Fraud"], digits=4))

    top_k = recall_at_top_k(y_test, test_scores)
    logger.info("Frauds caught when reviewing only the riskiest transactions:")
    for name, row in top_k.items():
        logger.info("  %-10s review %6d txns -> %3d / %d frauds (recall %.1f%%)",
                    name, row["reviewed"], row["frauds_caught"],
                    int(y_test.sum()), row["recall"] * 100)

    # Persist artefacts for the report.
    history = model.loss_history_
    save_plots(y_test, test_scores, model.threshold_, history, metrics,
               cfg.output_dir)
    summary = {"config": {k: str(v) if isinstance(v, Path) else v
                          for k, v in asdict(cfg).items()},
               "metrics": metrics, "recall_at_top_k": top_k,
               "threshold": float(model.threshold_)}
    with open(cfg.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("=" * 64)
    logger.info("Saved plots and metrics.json to '%s/'", cfg.output_dir)
    return summary


def parse_args() -> ExperimentConfig:
    """Read command-line options into an ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="Unsupervised fraud detection with a PyOD AutoEncoder.")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"),
                        help="Path to the Kaggle creditcard.csv file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Folder for plots, metrics and the run log.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size.")
    args = parser.parse_args()
    return ExperimentConfig(data_path=args.data, output_dir=args.output_dir,
                            epochs=args.epochs, batch_size=args.batch_size)


def main() -> int:
    """Entry point: returns a process exit code."""
    cfg = parse_args()
    configure_logging(cfg.output_dir / "run_log.txt")
    try:
        run(cfg)
    except (FileNotFoundError, ValueError) as err:
        logger.error(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
